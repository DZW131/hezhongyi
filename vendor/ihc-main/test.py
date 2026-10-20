import os
import pdb
import sys
import argparse
import random
import wandb
import cv2 as cv
# from projects.pp_mobileseg.inference_onnx import visualize

from utils import *
from tqdm import tqdm

from dataset import build_dataset
from models.detr_p2p.detr import build_model
# from dpa_models.dpa_p2pnet import build_model
# from models.dpa_p2p.dpa_p2pnet import build_dpa_model as build_model
from loss import build_criterion

from matcher import build_matcher
from lr_sched import adjust_learning_rate

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

class Evaluator:
    def __init__(self, args, data_loader_val):
        super(Evaluator, self).__init__()
        self.data_loader = data_loader_val
        # print('data_loader dataset', self.data_loader.dataset.data)
        self.gds = []
        for sample in self.data_loader.dataset.data:
            self.gds.append(tuple(sample.values())[1:])
        self.num_classes = args.num_classes

    @torch.no_grad()
    def predict(self, model, images, apply_deduplication: bool = False):
        h, w = images.shape[-2:]
        outputs = model(images)

        points = outputs['pnt_coords'][0].cpu().numpy()
        scores = torch.softmax(outputs['cls_logits'][0], dim=-1).cpu().numpy()

        cross_border_index = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
        points = points[~cross_border_index]
        scores = scores[~cross_border_index]

        classes = np.argmax(scores, axis=-1)
        reserved_index = classes < args.num_classes

        if apply_deduplication:
            return deduplicate(points[reserved_index], scores[reserved_index], 12)
        else:
            return points[reserved_index], classes[reserved_index]

    def calculate_metrics(self, model, effective_matching_dis=12, rank=0):
        eps = 1e-8
        model.eval()

        cls_pn, cls_tn, cls_rn = list(torch.zeros(self.num_classes).cuda(rank) for _ in range(3))
        det_rn, det_tn, det_pn = list(torch.zeros(1).cuda(rank) for _ in range(3))
        for i, (images, points, labels, lengths) in enumerate(self.data_loader):

            images = images.cuda(non_blocking=True)
            pd_points, pd_classes = self.predict(model, images, apply_deduplication=True)
            gd_points = np.zeros((0, 2), dtype=int)
            for c in range(self.num_classes):
                category_pd_points = pd_points[pd_classes == c]
                # print('gds see', self.gds)
                category_gd_points = self.gds[i][c]

                gd_points = np.concatenate([gd_points, category_gd_points], axis=0)

                pred_num, gd_num = len(category_pd_points), len(category_gd_points)

                cls_pn[c] += pred_num
                cls_tn[c] += gd_num

                if pred_num and gd_num:
                    right_num, _ = binary_match(category_pd_points, category_gd_points,
                                                threshold_distance=effective_matching_dis)
                    cls_rn[c] += right_num

            det_pn += len(pd_points)
            det_tn += len(gd_points)

            if len(pd_points) and len(gd_points):
                right_num, _ = binary_match(pd_points, gd_points, threshold_distance=effective_matching_dis)
                det_rn += right_num


        det_r = (det_rn + eps) / (det_tn + eps)
        det_p = (det_rn + eps) / (det_pn + eps)
        det_f1 = (2 * det_r * det_p + eps) / (det_p + det_r + eps)

        cls_r = (cls_rn + eps)/ (cls_tn + eps)
        cls_p = (cls_rn + eps)/ (cls_pn + eps)
        cls_f1 = (2 * cls_r * cls_p + eps) / (cls_r + cls_p + eps)

        metrics = {'检测指标': [det_p.item(), det_r.item(), det_f1.item()],
                   '分类精度': cls_p.tolist(), '分类召回': cls_r.tolist(), '分类F1': cls_f1.tolist(),
                   '分类指标': [cls_p.mean().item(), cls_r.mean().item(), cls_f1.mean().item()]}
        return metrics

    def visual_analysis(self, model, visualize_dir='vis_results/normal'):
        from skimage import io

        model.eval()
        os.makedirs(visualize_dir, exist_ok=True)
        # if not os.path.exists(output_dir):
        #     os.mkdir(output_dir)

        # colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]  # PDL1
        # colors = [(102, 255, 0), (255, 0, 0), (234, 255, 0), (0, 238, 255)]  # BXR
        colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 204, 153), (255, 153, 255), # PDL1_all
                  (204, 0, 255), (153, 255, 204), (0, 255, 255), (204, 153, 0)]

        for i, (images, points, labels, lengths) in enumerate(self.data_loader):
            img_name = self.data_loader.dataset.files[i].split('.')[0]
            print(f'Visualizing --- {img_name}')

            images = images.cuda(non_blocking=True)

            pd_points, pd_classes = self.predict(model, images, apply_deduplication=True)

            # draw pred points
            image = self.data_loader.dataset.data[i]['image'].copy()
            image1 = image.copy()
            for c, (x, y) in zip(pd_classes.astype(int), pd_points.astype(int)):
                cv.line(image, (x - 6, y), (x + 6, y), color=colors[c], thickness=4, lineType=cv.LINE_AA)
                cv.line(image, (x, y - 6), (x, y + 6), color=colors[c], thickness=4, lineType=cv.LINE_AA)
            # io.imsave(f"{output_dir}/{img_name}_pred.png", image, check_contrast=False)

            # draw gd points
            for c in range(self.num_classes):
                for (x, y) in self.gds[i][c].astype(int):
                    cv.circle(image, (x, y), radius=12, color=colors[c], thickness=1, lineType=cv.LINE_AA)
                    # cv.line(image1, (x - 6, y), (x + 6, y), color=colors[c], thickness=4, lineType=cv.LINE_AA)
                    # cv.line(image1, (x, y - 6), (x, y + 6), color=colors[c], thickness=4, lineType=cv.LINE_AA)
            # io.imsave(f"{output_dir}/{img_name}_gd.png", image1, check_contrast=False)

            io.imsave(f"{visualize_dir}/{img_name}.png", image, check_contrast=False)

    @torch.no_grad()
    def match_procedure(self, model):
        model.eval()
        matcher = build_matcher(args)
        for i, (images, points, labels, lengths) in enumerate(self.data_loader):
            img_name = self.data_loader.dataset.files[i].split('.')[0]
            print(f'Processing --- ({img_name})')

            images = images.cuda(non_blocking=True)
            points = points.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)

            targets = {'gt_nums': lengths,
                       'gt_points': [points_seq[points_seq != -1].reshape(-1, 2) for points_seq in points],
                       'gt_labels': [label_seq[label_seq != -1] for label_seq in labels]}

            indices = matcher(outputs, targets)
            np.save(f'./tool/result/{img_name}_indices',
                    [indices[0][0].cpu().numpy(), indices[0][1].cpu().numpy()])

            cls_scores = outputs['cls_logits'][0].softmax(-1).cpu().numpy()
            points = outputs['pnt_coords'][0].cpu().numpy()

            reg_attn = outputs['reg_attn'][0, ..., 0].cpu().numpy()
            cls_attn = outputs['cls_attn'][0, ..., 0].cpu().numpy()

            np.save(f'./tool/result/{img_name}_scores', cls_scores)
            np.save(f'./tool/result/{img_name}_points', points.astype(int))
            np.save(f'./tool/result/{img_name}_cls_attn', reg_attn)
            np.save(f'./tool/result/{img_name}_reg_attn', cls_attn)


def eval(args):
    model = build_model(args, args.num_classes)
    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(ckpt['epoch'], ckpt['metrics'])
    # ckpt = torch.load(f'./{ckpt_path}', map_location='cpu')

    dataset_val = build_dataset(args, is_train=False)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate_fn_pad)

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt['model'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()

    evaluator = Evaluator(args, data_loader_val)
    metrics = evaluator.calculate_metrics(model, effective_matching_dis=args.match_dis)
    print(metrics)
    # pdb.set_trace()

    evaluator.visual_analysis(model, visualize_dir=args.visualize_dir)
    # evaluator.match_procedure(model)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )
    parser.add_argument('--num_classes', type=int, default=4,
                        help="Number of cell categories")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--model', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")

    parser.add_argument('--add_tf_encoder', default=False, type=bool,
                        help="add_tf_encoder or not")
    parser.add_argument('--num_levels', default=3, type=int,
                        help="level number")
    parser.add_argument('--num_fov', default=0, type=int,
                        help="fov number")
    parser.add_argument('--ssl_type', default='', type=str)
    parser.add_argument('--ssl_path', default='', type=str)

    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--space', default=8, type=int,
                        help="AnchorPoints")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--row', default=2, type=int, help="number of anchor points per row")
    parser.add_argument('--col', default=2, type=int, help="number of anchor points per column")
    parser.add_argument('--grid', default=32, type=int, help="number of anchor points per column")


    parser.add_argument('--match_dis', default=12, type=int)
    # * Dataset
    parser.add_argument('--dataset', default='pdl1', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pre_load', action='store_true', help='pre-load dataset')

    # * Checkpoint
    parser.add_argument('--ckpt_path', default='pdl1', type=str)

    # * Visualize Path
    parser.add_argument('--visualize_dir', default='pdl1', type=str)

    # parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    cudnn.benchmark = True

    eval(args)

    # cls_mf1 = sum(metrics['分类F1'][:2]) / 2
    # print(metrics, cls_mf1)