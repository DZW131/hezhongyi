import torch
import numpy as np
import torch.nn as nn

class AnchorPoints(nn.Module):
    def __init__(self, space=16):
        super(AnchorPoints, self).__init__()
        self.space = space

    def forward(self, images):
        bs, _, h, w = images.shape
        anchors = np.stack(
            np.meshgrid(
                np.arange(np.ceil(w / self.space)),
                np.arange(np.ceil(h / self.space))),
            -1) * self.space

        origin_coord = np.array([w % self.space or self.space, h % self.space or self.space]) / 2
        anchors += origin_coord

        anchors = torch.from_numpy(anchors).float().to(images.device)
        return anchors.repeat(bs, 1, 1, 1)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, drop=0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList()

        for n, k in zip([input_dim] + h, h):
            self.layers.append(nn.Linear(n, k))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(drop))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
