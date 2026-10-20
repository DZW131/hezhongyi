import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F



class Crierition(nn.Module):
    def __init__(self, num_classes, matcher, class_weight, loss_weight):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.class_weight = class_weight

    def loss_reg(self, outputs, targets, indices, num_points):
        """ Regression loss """
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pnt_coords'][idx]

        target_points = torch.cat([gt_points[J] for gt_points, (_, J) in zip(targets['gt_points'], indices)], dim=0)
        if len(target_points) == 0:
            return {'loss_reg': torch.tensor(0).to(src_points)}

        loss_pnt = F.mse_loss(src_points, target_points, reduction='none')
        loss_dict = {'loss_reg': loss_pnt.sum() / num_points}
        return loss_dict

    def loss_cls(self, outputs, targets, indices, num_points):
        """Classification loss """
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['cls_logits']

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.long, device=src_logits.device)
        target_classes_o = torch.cat([cls[J] for cls, (_, J) in zip(targets['gt_labels'], indices)])
        target_classes[idx] = target_classes_o

        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.class_weight)
        loss_dict = {'loss_cls': loss_cls}
        return loss_dict

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        indices = self.matcher(outputs, targets)

        num_points = sum(targets['gt_nums'])
        num_points = torch.as_tensor(num_points, dtype=torch.float)

        losses = {}
        loss_map = {
            'loss_reg': self.loss_reg,
            'loss_cls': self.loss_cls,
        }

        for loss_func in loss_map.values():
            losses.update(loss_func(outputs, targets, indices, num_points))

        weight_dict = self.loss_weight
        losses = torch.stack([losses[k] * weight_dict[k] for k in weight_dict if k in losses])
        return losses


def build_criterion(rank, matcher, args):
    class_weight = torch.ones(args.num_classes + 1, dtype=torch.float, device=f'cuda:{rank}')
    #class_weight[0:-1] = class_weight_np
    class_weight[-1] = args.eos_coef

    loss_weight = {'loss_reg': args.reg_loss_coef, 'loss_cls': args.cls_loss_coef}
    return Crierition(
        args.num_classes,
        matcher,
        class_weight=class_weight,
        loss_weight=loss_weight
    )
