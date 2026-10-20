"""
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network"""

    def __init__(self, cost_point, cost_class):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.top_k = 1

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        修改说明：增加对空 Batch 的兼容处理，防止 cdist 报错
        """
        bs, num_queries = outputs["pnt_coords"].shape[:2]

        # 1. 先把所有真值点拼起来
        # targets['gt_points'] 是一个 list，包含 bs 个 tensor
        tgt_points = torch.cat(targets['gt_points'])

        # 2. 【核心修改】检查是否为空
        # 如果当前 batch 所有图片都没有真值点 (shape[0] == 0)，直接返回空匹配
        if tgt_points.shape[0] == 0:
            # 返回格式必须是 list of tuple: [(pred_idx, gt_idx), ...]
            # pred_idx 和 gt_idx 都是空的 int64 tensor
            return [(torch.as_tensor([], dtype=torch.int64), 
                     torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]

        # --- 以下是正常的计算逻辑 (只有当存在真值时才执行) ---

        # Compute the regression cost.
        out_coords = outputs["pnt_coords"].flatten(0, 1) # [bs * num_queries, 2]
        
        # 此时 tgt_points 肯定不为空，可以安全计算 cdist
        cost_point = torch.cdist(out_coords, tgt_points, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        # cost_point = torch.cdist(out_coords, tgt_points, p=2)

        # Compute the classification cost.
        out_prob = outputs["cls_logits"].flatten(0, 1).softmax(-1)
        
        # 同理，tgt_labels 拼接
        tgt_labels = torch.cat(targets['gt_labels'])
        
        # 这里的 indexing 可能会消耗显存，但在 point 不为空时是安全的
        cost_class = - (out_prob[:, tgt_labels] + 1e-8).log()

        # Final cost matrix.
        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).detach().cpu()

        # 根据每张图的 gt 数量进行切分
        # indices[i] = (row_ind, col_ind)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(targets['gt_nums'], -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    # assign 函数似乎没被调用，保留原样或按需删除均可
    @torch.no_grad()
    def assign(self, pred_point, pred_prob, gt_points, gt_labels, eps=1e-8):
        # 如果你想让这个函数也兼容，这里也得加判断，但上面的 forward 已经拦截了空数据
        if len(gt_points) == 0:
            return (torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64))

        cost_point = torch.cdist(pred_point, gt_points, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        cost_class = - pred_prob[:, gt_labels]

        cost = self.cost_point * cost_point + self.cost_class * cost_class
        cost = cost.cpu().numpy()

        n, m = len(pred_point), len(gt_points)
        index = torch.arange(n)
        valid_flag = torch.ones((n,), dtype=torch.bool)
        assign_matrix = torch.zeros((n, m), dtype=torch.bool)

        for _ in range(self.top_k):
            new_cost = cost[valid_flag]
            new_index = index[valid_flag]

            matched_row_inds, matched_col_inds = linear_sum_assignment(new_cost)
            matched_row_inds = torch.from_numpy(matched_row_inds)
            matched_col_inds = torch.from_numpy(matched_col_inds)

            matched_row_inds = new_index[matched_row_inds]  # convert to global index

            assign_matrix[matched_row_inds, matched_col_inds] = True
            valid_flag[matched_row_inds] = False

        return torch.where(assign_matrix)


def build_matcher(args):
    matcher = HungarianMatcher(
        cost_point=args.set_cost_point,
        cost_class=args.set_cost_class
    )
    return matcher