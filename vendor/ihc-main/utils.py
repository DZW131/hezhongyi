import os

import torch
import torch.distributed as dist

import numpy as np
import scipy.spatial as S

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from torch.nn.utils.rnn import pad_sequence


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    print(args.rank,args.world_size)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def cleanup():
    dist.destroy_process_group()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def binary_match(pred_points, gd_points, threshold_distance=12):
    dis = S.distance_matrix(pred_points, gd_points)
    connection = np.zeros_like(dis)
    connection[dis < threshold_distance] = 1
    graph = csr_matrix(connection)
    res = maximum_bipartite_matching(graph, perm_type='column')
    # scipy encodes an unmatched left vertex as -1; index 0 is a valid match.
    right_points_index = np.where(res >= 0)[0]
    right_num = len(right_points_index)

    matched_gt_points = res[right_points_index]

    if len(np.unique(matched_gt_points)) != len(matched_gt_points):
        import pdb;
        pdb.set_trace()

    return right_num, right_points_index


def deduplicate(points, scores, interval):
    n = len(points)
    fused = np.full(n, False)
    result = np.zeros((0, 2))
    classes = np.array([])
    for i in range(n):
        if not fused[i]:
            fused_index = np.where(np.linalg.norm(points[[i]] - points[i:], 2, axis=1) < interval)[0] + i
            fused[fused_index] = True

            r_, c_ = np.where(scores[fused_index] == np.max(scores[fused_index]))
            r_, c_ = [r_[0]], [c_[0]]
            result = np.append(result, points[fused_index[r_]], axis=0)
            classes = np.append(classes, c_)
    return result, classes


def load_checkpoint(args, model, optimizer, evaluator=None):
    checkpoint = torch.load(f'./checkpoint/{args.resume}', map_location='cpu')

    # model_dict = model.state_dict()

    # # freeze network
    # # for name, param in checkpoint['model'].items():
    # #     param.requires_grad = False

    # pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    if 'optimizer' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    metrics = {}
    if evaluator is not None:
        metrics.update(evaluator.calculate_metrics(model))
    return metrics


def trigger_eval(epoch, start_eval, interval=1, total_epochs=None):
    if total_epochs is not None and epoch == total_epochs - 1:
        return True
    return epoch > start_eval and (epoch - start_eval) % max(interval, 1) == 0


def save_model(epoch, args, metrics, model, optimizer, is_recent=True):
    os.makedirs(f'./checkpoint/{args.output_dir}', exist_ok=True)
    if is_recent:
        torch.save({
            'epoch': epoch,
            'metrics': metrics,
            'structure': str(model),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f'./checkpoint/{args.output_dir}/recent.pth')
    else:
        torch.save({
            'epoch': epoch,
            'metrics': metrics,
            'structure': str(model),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f'./checkpoint/{args.output_dir}/best.pth')

def strip_model(args):
    checkpoint = torch.load(f'./checkpoint/{args.output_dir}/best.pth', map_location='cpu')
    torch.save({
        'model': checkpoint['model'],
    }, f'./checkpoint/{args.output_dir}.pth')


def collate_fn_pad(batch):
    batch.sort(key=lambda x: len(x[2]), reverse=True)  # sort by the number of points
    images, points, labels, lengths = [[] for _ in range(4)]
    for x in batch:
        images.append(x[0])
        points.append(x[1])
        labels.append(x[2])
        lengths.append(len(x[2]))
    points = pad_sequence(points, batch_first=True, padding_value=-1).reshape(len(batch), -1)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1).reshape(len(batch), -1)
    return torch.stack(images), points.float(), labels.long(), lengths


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def _draw_confusion_matrix():
    # ins = torch.load('hcn_59_best.pth')

    # peripheral_blood_dict = {"0": 'BAS,Basophil', "1": 'EBO,Erythroblast', "2": 'EOS,Eosinophil',
    #                          "3": 'KSC,Smudge cell', "4": 'LYA,Lymphocyte (atypical)',
    #                          "5": 'LYT,Lymphocyte (typical)', "6": 'MMZ,Metamyelocyte', "7": 'MOB,Monoblast',
    #                          "8": 'MON,Monocyte',
    #                          "9": 'MYB,Myelocyte', "10": 'MYO,Myeloblast', "11": 'NGB,Neutrophil (band)',
    #                          "12": 'NGS,Neutrophil (segmented)',
    #                          "13": 'PMB,Promyelocyte (bilobled)', "14": 'PMO,Promyelocyte'}
    # # array = ins['metrics'].to_array(normalized=True)
    # # np.save('array', array)
    # array = np.load('array.npy')
    #
    import seaborn as sn
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams["font.family"] = "Times New Roman"
    plt.tight_layout()
    # classes = [cls[4:] for cls in peripheral_blood_dict.values()]

    array = np.array([[2, 1, 5, 0, 0, 0, 0, 0],
                      [2, 36, 19, 7, 4, 2, 0, 0],
                      [1, 15, 84, 21, 6, 1, 2, 0],
                      [1, 11, 25, 36, 16, 7, 1, 1],
                      [2, 5, 23, 12, 30, 3, 3, 0],
                      [0, 1, 3, 7, 5, 15, 1, 1],
                      [0, 1, 3, 1, 2, 1, 10, 0],
                      [1, 1, 3, 2, 3, 0, 0, 3]])

    # df_cm = pd.DataFrame(array, index=classes, columns=classes)
    df_cm = pd.DataFrame(array)
    fig, ax = plt.subplots(figsize=(42, 42))

    res = sn.heatmap(df_cm,
                     square=True,
                     vmin=0, vmax=1,
                     ax=ax,
                     cmap='Blues',
                     cbar_kws={"shrink": 0.8})

    res.axhline(y=0, color='k', linewidth=8)
    res.axhline(y=array.shape[1], color='k', linewidth=8)
    res.axvline(x=0, color='k', linewidth=8)
    res.axvline(x=array.shape[0], color='k', linewidth=8)

    ax.figure.axes[-1].tick_params(labelsize=48)

    ax.tick_params(labelsize=48, direction='in', length=20, width=6)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    fig.savefig('sow.png', bbox_inches='tight')


def _draw_score_changes():
    # np.save("cls_scores.npy", cls_scores)
    categories = ['阴性肿瘤细胞', '阴性非肿瘤细胞', '阳性肿瘤细胞', '阳性非肿瘤细胞']
    classes_name = ['neg_tumor', 'neg_non_tumor', 'pos_tumor', 'pos_non_tumor']
    ylabels = ['negative tumor cell', 'negative non-tumor cell', 'positive tumor cell', 'pos non-tumor cell']

    cls_scores = np.load("cls_scores.npy")
    import matplotlib.pyplot as plt
    plt.rc('font', family='Helvetica')

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, 10) / 10
    ax.plot(x, cls_scores, marker='o', markersize=9, c='#541BA5')

    for i, score in zip(x, cls_scores):
        ax.annotate(str(round(score, 4)), xy=(i + 0.02, score - 0.0003), fontsize=16)

    ax.set_xlabel('$q$', fontsize=24)
    ax.set_ylabel('Average F1 score', fontsize=24)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(direction="in")

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(axis='y')
    fig.savefig('cls_scores.png', bbox_inches='tight')
