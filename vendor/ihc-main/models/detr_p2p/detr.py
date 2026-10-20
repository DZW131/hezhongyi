# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import copy

import numpy as np
import torch.nn.functional as F

from torch import nn
from .backbone import build_backbone


class AnchorPoints(nn.Module):
    def __init__(self, row=2, col=2, grid_scale=(32, 32)):
        super(AnchorPoints, self).__init__()
        x_space = grid_scale[0] / row
        y_space = grid_scale[1] / col
        self.deltas = np.array(
            [
                [-x_space, -y_space],
                [x_space, -y_space],
                [0, 0],
                [-x_space, y_space],
                [x_space, y_space]
            ]
        ) / 2

        self.grid_scale = np.array(grid_scale)

    def forward(self, images):
        bs, _, h, w = images.shape
        device = images.device
        dtype = images.dtype
        
        # 使用torch操作替代numpy操作以确保ONNX导出正确
        grid_scale_tensor = torch.tensor(self.grid_scale, device=device, dtype=dtype)
        deltas_tensor = torch.tensor(self.deltas, device=device, dtype=dtype)
        
        # 计算grid范围
        n_x = int(np.ceil(w / self.grid_scale[0]))
        n_y = int(np.ceil(h / self.grid_scale[1]))
        
        x_range = (torch.arange(n_x, device=device, dtype=dtype) + 0.5) * self.grid_scale[0]
        y_range = (torch.arange(n_y, device=device, dtype=dtype) + 0.5) * self.grid_scale[1]
        
        # 生成meshgrid
        yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
        centers = torch.stack([xx, yy], dim=-1)
        
        # 生成anchors
        anchors = centers.unsqueeze(2) + deltas_tensor
        anchors = anchors.flatten(0, 2).unsqueeze(0).repeat(bs, 1, 1)
        
        return anchors


class NeighborPointsGenerator(nn.Module):
    def __init__(self, radius=5, stride=4, base_point_num=8):
        super(NeighborPointsGenerator, self).__init__()

        self.radius = radius
        self.stride = stride
        self.base_point_num = base_point_num

        self.offset_x, self.offset_y = self.get_deltas()

    def forward(self, images, gt_points, gt_nums):
        pixels, flag = self.get_negative_flag(images, gt_points, gt_nums)
        flag = self.get_positive_flag(images, flag, gt_points)
        return pixels, flag

    def get_deltas(self):
        dx = torch.empty((0, 2))
        dy = torch.empty((0, 2))

        for i in range(self.radius):
            r = (i + 1) * self.stride
            num_pts = self.base_point_num * (i + 1)
            angles = torch.arange(num_pts) / num_pts * np.pi * 2

            dx = torch.cat([dx, r * torch.cos(angles)], 0)
            dy = torch.cat([dy, r * torch.sin(angles)], 0)

        return dx, dy

    def get_positive_flag(self, images, flag, gt_points):
        bs, _, h, w = images.shape
        max_index = h * w
        offset_index = self.offset_y * w + self.offset_x * h
        for i in range(bs):
            gt_points_index = (gt_points[i][:, 0] * w + gt_points[i][:, 1] * h) + offset_index
            gt_points_index = gt_points_index[gt_points_index >= 0 & gt_points_index < max_index]
            flag[i][gt_points_index] = 1
        return flag

    def get_negative_flag(self, images, gt_points, gt_nums):
        bs, _, h, w = images.shape

        pixels = torch.stack(
            torch.meshgrid(
                torch.arange(w),
                torch.arange(h)),
            -1).flatten(0, 1).to(gt_points.device)

        dis_thr = self.stride * self.radius
        dis_matrix = torch.cdist(pixels, torch.cat(gt_points), p=2)
        flag = torch.stack([sub_matrix.min(1) > dis_thr for sub_matrix in dis_matrix.split(gt_nums, 1)], 0)

        return pixels.repeat(bs, 1, 1), flag.float() * -1


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, hidden_dim, num_classes, row, col, grid_scale):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            num_classes: number of object classes
        """
        super().__init__()
        self.backbone = backbone
        self.get_aps = AnchorPoints(row, col, grid_scale)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_levels = self.backbone.neck.num_outs
        self.strides = [2 ** (i + 1) for i in range(self.num_levels)]

        # deformable roi pooling
        self.deformable_mlps = _get_clones(MLP(hidden_dim, hidden_dim, 2, 2), self.num_levels)
        for mlp in self.deformable_mlps:
            nn.init.constant_(mlp.layers[-1].bias.data, 0)
            nn.init.constant_(mlp.layers[-1].weight.data, 0)

        self.reg_head = MLP(hidden_dim, hidden_dim, 2, 3)
        self.cls_head = nn.Linear(hidden_dim, num_classes + 1)

        self.loc_aggr = nn.Sequential(nn.Linear(hidden_dim * self.num_levels, hidden_dim), nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, self.num_levels))
        self.cls_aggr = nn.Sequential(nn.Linear(hidden_dim * self.num_levels, hidden_dim), nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, self.num_levels))

    def forward(self, images):
        anchors = self.get_aps(images)
        features = self.backbone(images)

        reg_features, cls_features, reg_attn, cls_attn = self.extract_features(features, anchors)
        points = self.reg_head(reg_features) + anchors
        logits = self.cls_head(cls_features)
        return points, logits

    def extract_features(self, features, points, align_corners=True):
        # 使用.to()替代.cuda()以确保ONNX导出正确，并且明确指定dtype
        roi_features = torch.zeros(self.num_levels, *points.shape[:2], self.hidden_dim, 
                                   dtype=points.dtype, device=points.device)
        for i, stride in enumerate(self.strides):
            h, w = features[i].shape[2:]
            scale = torch.tensor([w, h], dtype=points.dtype, device=points.device)
            grid = (2.0 * points / stride / scale - 1.0).unsqueeze(2)  # for alignment

            pre_roi_features = F.grid_sample(features[i], grid, align_corners=align_corners).squeeze(-1).permute(0, 2,
                                                                                                                 1)
            grid = (2.0 * (points + self.deformable_mlps[i](pre_roi_features)) / stride / scale - 1.0).unsqueeze(2)

            roi_features[i] = F.grid_sample(features[i], grid, align_corners=align_corners).squeeze(-1).permute(0, 2, 1)

        roi_features = roi_features.permute(1, 2, 0, 3)
        attn_features = roi_features.flatten(2)

        reg_attn = F.softmax(self.loc_aggr(attn_features), dim=-1).unsqueeze(-1)
        reg_features = (reg_attn * roi_features).sum(dim=2)

        cls_attn = F.softmax(self.cls_aggr(attn_features), dim=-1).unsqueeze(-1)
        cls_features = (cls_attn * roi_features).sum(dim=2)

        return reg_features, cls_features, reg_attn, cls_attn


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_model(args, num_classes):
    backbone = build_backbone(args)
    grid_scale = (args.grid, args.grid)

    model = DETR(
        backbone,
        row=args.row,
        col=args.col,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        grid_scale=grid_scale
    )

    return model
