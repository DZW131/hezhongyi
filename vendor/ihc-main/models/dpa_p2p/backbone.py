# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import copy
import timm
import torch

from torch import nn
from .fpn import FPN
from functools import reduce
from .utils import load_vit_weights
from .vit_adapter import ViTAdapter

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
            self,
            args
    ):
        super(Backbone, self).__init__()

        backbone = ViTAdapter(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            drop_path_rate=0.2,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=12,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            use_extra_extractor=False,
            layer_scale=False,
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
            window_attn=[True, True, False, True, True, False,
                         True, True, False, True, True, False],
            window_size=[14, 14, None, 14, 14, None,
                         14, 14, None, 14, 14, None])

        # IN pretrain
        load_vit_weights(backbone, ssl_type=args.ssl_type, ssl_path=args.ssl_path)

        self.backbone = backbone
        self.backbone_name = args.backbone

        if args.backbone not in ['vitdet', 'sam']:
            tf_encoder = None

            if args.backbone.startswith('resnet50'):
                in_channels = [256, 512, 1024, 2048]
            elif args.backbone.startswith('convnext'):
                # in_channels = [128, 256, 512, 1024]
                in_channels = [96, 192, 384, 768]
            else:
                in_channels = [768, 768, 768, 768]
                
            self.neck = FPN(
                in_channels=in_channels,
                out_channels=args.hidden_dim,
                num_outs=args.num_levels,
                add_extra_convs='on_input'
            )

        if args.num_fov > 0:
            self.fpn_convs = nn.ModuleList()

            for i in range(args.num_levels):
                self.fpn_convs.append(
                    nn.Conv2d(args.hidden_dim, args.hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            self.ups = nn.ModuleList(
                [nn.ConvTranspose2d(256, 256, kernel_size=2 ** i, stride=2 ** i) for i in
                 range(1, args.num_fov + 1)])

            # self.ups = nn.ModuleList(
            #     [nn.ConvTranspose2d(256, 256, kernel_size=4, stride=4)])

            for i in range(1, args.num_fov + 1):
                setattr(self, f'backbone{i}', copy.deepcopy(self.backbone))
                setattr(self, f'neck{i}', copy.deepcopy(self.neck))

    def forward(self, images):
        x = self.backbone(images[:, :3])[-4:]

        if hasattr(self, 'neck'):
            out = list(self.neck(x))
        else:
            out = list(x.values())
            # out = list(x)

        if hasattr(self, 'fpn_convs'):
            for i, j in enumerate(range(3, images.size(1), 3)):

                if hasattr(self, 'backbone1'):
                    fov_out = getattr(self, f'backbone{i + 1}')(images[:, j:j + 3])
                    fov_out = list(getattr(self, f'neck{i + 1}')(fov_out))
                else:
                    fov_out = self.backbone(images[:, j:j + 3])
                    if hasattr(self, 'neck'):
                        fov_out = list(self.neck(fov_out))

                for k in range(len(fov_out)):
                    coef = i + 1
                    h, w = fov_out[k].shape[-2:]
                    r1, r2 = int((2 ** coef - 1) / 2 ** (coef + 1) * h), int((2 ** coef + 1) / 2 ** (coef + 1) * h)
                    c1, c2 = int((2 ** coef - 1) / 2 ** (coef + 1) * w), int((2 ** coef + 1) / 2 ** (coef + 1) * w)

                    # x_offset, y_offset = offsets[i]
                    # r1, r2 = int(y_offset * h - h // 8), int(y_offset * h + h // 8)
                    # c1, c2 = int(x_offset * w - w // 8), int(x_offset * w + w // 8)

                    out[k] = out[k] + self.ups[i](fov_out[k][..., r1:r2, c1:c2])

            for i in range(len(out)):
                out[i] = self.fpn_convs[i](out[i])

        return out


def build_backbone(args):
    backbone = Backbone(args)
    return backbone


def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')[:-1]
    return reduce(getattr, names, module)
