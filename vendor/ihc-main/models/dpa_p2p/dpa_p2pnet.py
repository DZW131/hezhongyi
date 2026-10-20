# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn
from .backbone import build_backbone
from .modules import AnchorPoints, MLP


class DPAP2PNet(nn.Module):
    """ This is the Proposal-aware P2PNet module that performs cell recognition """

    def __init__(
            self,
            backbone,
            num_levels,
            num_classes,
            dropout=0.1,
            space: int = 16,
            hidden_dim: int = 256
    ):
        """
            Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.get_aps = AnchorPoints(space)
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.strides = [2 ** (i + 2) for i in range(self.num_levels)]

        self.deform_layer = MLP(hidden_dim, hidden_dim, 2, 2, drop=dropout)

        self.reg_head = MLP(hidden_dim * num_levels, hidden_dim, 2, 2, drop=dropout)
        self.cls_head = MLP(hidden_dim * num_levels, hidden_dim, 2, num_classes + 1, drop=dropout)

    def forward(self,
                images):
        # extract features
        feats, proposals = self.backbone(images), self.get_aps(images)

        feat_sizes = [torch.tensor(feat.shape[:1:-1], dtype=torch.float, device=proposals.device) for feat in feats]

        # DPP
        grid = (2.0 * proposals / self.strides[0] / feat_sizes[0] - 1.0)
        roi_features = F.grid_sample(feats[0], grid, mode='bilinear', align_corners=True)
        deltas2deform = self.deform_layer(roi_features.permute(0, 2, 3, 1))
        deformed_proposals = proposals + deltas2deform

        # MSD
        roi_features = []
        for i in range(self.num_levels):
            grid = (2.0 * deformed_proposals / self.strides[i] / feat_sizes[i] - 1.0)
            roi_features.append(F.grid_sample(feats[i], grid, mode='bilinear', align_corners=True))
            # roi_features.append(self.grid_sample(feats[i], grid))
        roi_features = torch.cat(roi_features, 1).permute(0, 2, 3, 1)

        # Head
        deltas2refine = self.reg_head(roi_features)
        pred_coords = deformed_proposals + deltas2refine

        pred_logits = self.cls_head(roi_features)

        output = {
            'pnt_coords': pred_coords.flatten(1, 2),
            'cls_logits': pred_logits.flatten(1, 2)
        }

        return output


def build_dpa_model(args, num_classes):

    backbone = build_backbone(args)

    model = DPAP2PNet(
        backbone,
        args.num_levels,
        num_classes,
        dropout=args.dropout,
        space=args.space,
        hidden_dim=args.hidden_dim
    )

    return model
