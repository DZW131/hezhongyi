import os
import sys
import argparse
import random
import wandb
import cv2 as cv

from utils import *
from tqdm import tqdm

from dataset import build_dataset
from loss import build_criterion
from matcher import build_matcher
from lr_sched import adjust_learning_rate

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )

    # * Optimizer
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # * Train
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--start_eval', default=100, type=int)
    parser.add_argument('--eval_interval', default=1, type=int,
                        help='Evaluate every N epochs after start_eval; the final epoch is always evaluated.')
    parser.add_argument('--save_interval', default=1, type=int,
                        help='Write the rolling checkpoint every N epochs.')

    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--output_dir', default='', type=str,
                        help='DEPRECATED: Use --name instead. Path where to save, empty for no saving')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--num_classes', type=int, default=4,
                        help="Number of cell categories")

    # * Loss
    parser.add_argument('--reg_loss_coef', default=2e-3, type=float)
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=1.0, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--garbage_class_idx', default=-1, type=int,
                        help="Garbage class index, does not participate in training")

    # * Matcher
    parser.add_argument('--set_cost_point', default=0.1, type=float,
                        help="L2 point coefficient in the matching cost")
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--split_stride', default=1, type=int,
                        help="Split for train accelaerate")

    # * Model
    parser.add_argument('--model', default='', type=str,
                        help="Name of the model to use")
    
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--grid', default=32, type=int, help="anchor sparsity")
    parser.add_argument('--row', default=2, type=int, help="row number")
    parser.add_argument('--col', default=2, type=int, help="col number")

    # * Model
    parser.add_argument('--add_tf_encoder', default=False, type=bool,
                        help="add_tf_encoder or not")
    parser.add_argument('--num_levels', default=3, type=int,
                        help="level number")
    parser.add_argument('--num_fov', default=0, type=int,
                        help="fov number")
    parser.add_argument('--space', default=8, type=float,
                        help="Number of space for anchor proposal space")
    parser.add_argument('--ssl_type', default='', type=str)
    parser.add_argument('--ssl_path', default='', type=str)

    # * Evaluator
    parser.add_argument('--eval_only', default=False, type=bool)
    parser.add_argument('--match_dis', default=12, type=int)

    # * Distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--project',type=str,default='cell_recognation')
    parser.add_argument('--name',type=str,default='exp',
                        help='Experiment name, used as output directory name in checkpoint/ folder')

    parser.add_argument('--dataset', default='pdl1', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pre_load', action='store_true', help='pre-load dataset')


    return parser


def construct_dataset():
    dataset_train = build_dataset(args, is_train=True)
    dataset_val = build_dataset(args, is_train=False)

    if args.distributed:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        data_loader_train = DataLoader(dataset_train, sampler=train_sampler, batch_size=args.batch_size,
                                       num_workers=args.num_workers, collate_fn=collate_fn_pad, pin_memory=True)
    else:
        data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size,
                                       num_workers=args.num_workers, collate_fn=collate_fn_pad, pin_memory=True)
    data_loader_val = DataLoader(dataset_val, shuffle=False, batch_size=1, num_workers=args.num_workers,
                                 collate_fn=collate_fn_pad, pin_memory=True)

    data_loaders = {'train': data_loader_train, 'test': data_loader_val}
    return data_loaders


def train():
    if args.distributed:
        rank = args.gpu
    else:
        rank = 0

    if rank == 0:
        wandb.init(project=args.project, name=args.name, mode=os.getenv("WANDB_MODE", "disabled"))
        cfg = wandb.config
        for k, v in args.__dict__.items():
            setattr(cfg, k, v)

    if args.model == 'detr':
        from models.detr_p2p.detr import build_model
        model = build_model(args, num_classes=args.num_classes).cuda(rank)
    elif args.model == 'dpa':
        from models.dpa_p2p.dpa_p2pnet import build_dpa_model
        model = build_dpa_model(args, num_classes=args.num_classes).cuda(rank)
    else: 
        raise ValueError(f'the given model {args.model} is not implemented')
    model_without_ddp = model

    if args.distributed:
        if args.resume:
            model = DistributedDataParallel(model, device_ids=[rank], output_device=rank,find_unused_parameters=True)
        else:
            model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        model_without_ddp = model.module
    
    if args.resume:
        checkpoint = torch.load(f'./checkpoint/{args.resume}', map_location='cpu', weights_only=False)
        model_dict = model_without_ddp.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model_without_ddp.load_state_dict(model_dict)

        for name,param in model_without_ddp.named_parameters():
            if name.split('.')[0] not in ['cls_head','cls_aggr']:
                param.requires_grad = False
    
    matcher = build_matcher(args)
    criterion = build_criterion(rank, matcher, args)
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    data_loaders = construct_dataset()
    evaluator = Evaluator(data_loaders['test'])
    first_eval = True
    if args.eval_only:
        metrics = evaluator.calculate_metrics(model, rank=rank, effective_matching_dis=args.match_dis)
        print(metrics)
        return
    
    #return
    #metrics = load_checkpoint(args, model_without_ddp, optimizer) if args.resume else {}
    metrics = {}
    max_cls_mf1 = metrics.get('分类F1', 0)

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, data_loaders['train'], optimizer, args.clip_max_norm, epoch, criterion, rank)

        if trigger_eval(epoch, args.start_eval, args.eval_interval, args.epochs):
            if first_eval:
                save_model(epoch, args, metrics, model_without_ddp, optimizer,is_recent=False)
                first_eval = False

            metrics = evaluator.calculate_metrics(model, rank=rank, effective_matching_dis=args.match_dis)
            cls_mf1 = metrics['分类指标'][-1]
            print(metrics)

            if rank == 0:
                wandb.log(dict(zip(["检测精度", "检测召回", "检测F1"], metrics['检测指标'])))
                wandb.log(dict(zip(["分类精度", "分类召回", "分类F1"], metrics['分类指标'])))

                if max_cls_mf1 < cls_mf1:
                    max_cls_mf1 = cls_mf1
                    if args.output_dir:
                        save_model(epoch, args, metrics, model_without_ddp, optimizer,is_recent=False)
        
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_model(epoch,args,metrics, model_without_ddp,optimizer,is_recent=True)

    if rank == 0:
        strip_model(args)

    if args.distributed:
        cleanup()


def train_one_epoch(model, train_loader, optimizer, max_norm, epoch, criterion, rank):
    model.train()
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)

    iterator = train_loader
    if rank == 0:
        iterator = tqdm(train_loader, file=sys.stdout)
        iterator.set_description("Train epoch-%d" % epoch)

    reg_tl = cls_tl = 0
    for data_iter_step, (images, points, labels, lengths) in enumerate(iterator):
        # warmup lr
        adjust_learning_rate(args,
                             optimizer,
                             data_iter_step / len(iterator) + epoch,
                             rank)

        images = images.cuda(rank)
        points = points.cuda(rank)
        labels = labels.cuda(rank)

        # all points N×2
        targets = {'gt_nums': lengths,
                   'gt_points': [points_seq[points_seq != -1].reshape(-1, 2) for points_seq in points],
                   'gt_labels': [label_seq[label_seq != -1] for label_seq in labels]}

        outputs = model(images)
        losses = criterion(outputs, targets)
        loss = losses.sum()
        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:  # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        losses /= args.world_size
        if args.distributed:
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
        reg_tl += losses[0].item()
        cls_tl += losses[1].item()

    print(f'回归损失: {reg_tl}, 分类损失: {cls_tl}')
    if rank == 0:
        wandb.log({
            'reg_loss': reg_tl,
            'cls_loss': cls_tl,
            'tot_loss': reg_tl + cls_tl,
        })


class Evaluator:
    def __init__(self, data_loader_val):
        super(Evaluator, self).__init__()
        self.data_loader = data_loader_val
        self.gds = []
        for sample in data_loader_val.dataset.data:
            self.gds.append(tuple(sample.values())[1:])
        self.num_classes = args.num_classes

    @torch.no_grad()
    def predict(self, model, images, apply_deduplication: bool = False, match_distance=12):
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
            return deduplicate(points[reserved_index], scores[reserved_index], match_distance)
        else:
            return points[reserved_index], classes[reserved_index]

    def calculate_metrics(self, model, effective_matching_dis=12, rank=0):
        eps = 1e-8
        model.eval()

        cls_pn, cls_tn, cls_rn = list(torch.zeros(self.num_classes).cuda(rank) for _ in range(3))
        det_rn, det_tn, det_pn = list(torch.zeros(1).cuda(rank) for _ in range(3))
        for i, (images, points, labels, lengths) in enumerate(self.data_loader):

            if i % args.world_size != rank:
                continue

            images = images.cuda(non_blocking=True)
            pd_points, pd_classes = self.predict(model, images, apply_deduplication=True, match_distance=effective_matching_dis)
            gd_points = np.zeros((0, 2), dtype=int)
            for c in range(self.num_classes):
                
                category_pd_points = pd_points[pd_classes == c]
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

        if args.distributed:
            dist.all_reduce(det_rn, op=dist.ReduceOp.SUM)
            dist.all_reduce(det_tn, op=dist.ReduceOp.SUM)
            dist.all_reduce(det_pn, op=dist.ReduceOp.SUM)

            dist.all_reduce(cls_pn, op=dist.ReduceOp.SUM)
            dist.all_reduce(cls_tn, op=dist.ReduceOp.SUM)
            dist.all_reduce(cls_rn, op=dist.ReduceOp.SUM)

        # 检测指标计算
        # det_rn, det_tn, det_pn 是 shape=(1,) 的tensor
        det_rn_val = det_rn[0]
        det_tn_val = det_tn[0]
        det_pn_val = det_pn[0]
        
        if det_tn_val.item() == 0:
            det_r = torch.tensor(0.0).cuda(rank)
        else:
            det_r = det_rn_val / det_tn_val
        if det_pn_val.item() == 0:
            det_p = torch.tensor(0.0).cuda(rank)
        else:
            det_p = det_rn_val / det_pn_val
        if det_p.item() == 0 and det_r.item() == 0:
            det_f1 = torch.tensor(0.0).cuda(rank)
        else:
            det_f1 = 2 * det_r * det_p / (det_p + det_r + eps)

        # 分类指标计算（按类别）
        cls_r = torch.zeros(self.num_classes).cuda(rank)
        cls_p = torch.zeros(self.num_classes).cuda(rank)
        cls_f1 = torch.zeros(self.num_classes).cuda(rank)
        
        for c in range(self.num_classes):
            # Recall计算：当没有ground truth时，recall为0
            if cls_tn[c].item() == 0:
                cls_r[c] = 0.0
            else:
                cls_r[c] = cls_rn[c] / cls_tn[c]
            
            # Precision计算：当没有预测时，precision为0
            if cls_pn[c].item() == 0:
                cls_p[c] = 0.0
            else:
                cls_p[c] = cls_rn[c] / cls_pn[c]
            
            # F1计算：当precision和recall都为0时，F1为0；否则使用标准公式
            if cls_p[c].item() == 0 and cls_r[c].item() == 0:
                cls_f1[c] = 0.0
            elif cls_p[c].item() == 0 or cls_r[c].item() == 0:
                cls_f1[c] = 0.0
            else:
                cls_f1[c] = 2 * cls_r[c] * cls_p[c] / (cls_p[c] + cls_r[c] + eps)

        metrics = {'检测指标': [det_p.item(), det_r.item(), det_f1.item()],
                   '分类精度': cls_p.tolist(), '分类召回': cls_r.tolist(), '分类F1': cls_f1.tolist(),
                   '分类指标': [cls_p.mean().item(), cls_r.mean().item(), cls_f1.mean().item()]}
        return metrics



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    # 统一 output_dir 和 name：如果 output_dir 未指定，使用 name
    if not args.output_dir or args.output_dir == './output':
        args.output_dir = args.name
    # 如果两者不一致，给出警告并使用 name
    elif args.output_dir != args.name:
        print(f'Warning: --output_dir ({args.output_dir}) and --name ({args.name}) are different.')
        print(f'Using --name ({args.name}) as output directory. Please remove --output_dir parameter.')
        args.output_dir = args.name
    
    #os.environ['RANK'] = '0'
    #os.environ['WORLD_SIZE'] = '1'
    #os.environ['LOCAL_RANK'] = '0'
    #os.environ['MASTER_ADDR'] = '127.0.0.1'
    #os.environ['MASTER_PORT'] = '29500'

    init_distributed_mode(args)

    # fix the seed for reproducibility
    print('rank:',get_rank())
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    train()
