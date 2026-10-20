import torch
import numpy as np
import torch.nn as nn


class APGenerator(nn.Module):
    def __init__(self, row=2, col=2, grid_scale=(32, 32)):
        super(APGenerator, self).__init__()
        x_space = grid_scale[0] / row
        y_space = grid_scale[1] / col
        self.deltas = np.array(
            [[-x_space, -y_space],
             [x_space, -y_space],
             [0, 0],
             [-x_space, y_space],
             [x_space, y_space]]
        ) / 2

        self.grid_scale = np.array(grid_scale)

    def forward(self, samples):
        images = samples.tensors
        bs, _, h, w = images.shape
        anchors = np.stack(
            np.meshgrid(
                np.arange(np.ceil(w / self.grid_scale[0])) + 0.5,
                np.arange(np.ceil(h / self.grid_scale[1])) + 0.5),
            -1) * self.grid_scale
        all_anchors = np.expand_dims(anchors, 2) + self.deltas
        all_anchors = torch.from_numpy(all_anchors).float().to(images.device)
        return all_anchors.flatten(0, 2).repeat(bs, 1, 1)
