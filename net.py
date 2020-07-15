import torch
from torch import nn

def build_center_grid(height, width):
    shiftx = torch.arange(0, height, dtype=torch.float32) + 0.5
    shifty = torch.arange(0, width, dtype=torch.float32) + 0.5
    shifty, shiftx = torch.meshgrid(shiftx, shifty)
    xy_grid = torch.stack([shiftx, shifty], dim=-1)
    return xy_grid

class Decode(nn.Module):
    def __init__(self, num_classes, stride):
        super(Decode, self).__init__()
        self.num_classes = num_classes
        self.stride = stride
        self.xy_grid = build_center_grid(5, 10)

    def forward(self, conv):
        conv = conv.permute(0, 2, 3, 1)
        conv_raw_dxdy, conv_raw_prob = torch.split(conv, [2, self.num_classes], dim=-1)

        xy_grid = self.xy_grid.to(conv.device)
        pred_xy = (xy_grid[None, ...] + torch.sigmoid(conv_raw_dxdy) * 2 - 1) * self.stride
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred = torch.cat((pred_xy, pred_prob), -1)
        return pred

class Block(nn.Module):
    def __init__(self, w_in, w_out, stride, gw):
        super(Block, self).__init__()
        g = w_in // gw
        self.a = nn.Conv2d(w_in, w_in, 1, stride=1, padding=0, bias=False)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_in, w_in, 3, stride=stride, padding=1, groups=g, bias=False)
        self.b_relu = nn.ReLU(inplace=True)
        self.c = nn.Conv2d(w_in, w_out, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.add_module(
            'stem',
            nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
        )
        self.add_module('b1', Block(32, 32, 2, 8))
        self.add_module('b2', Block(32, 64, 2, 8))
        self.add_module('b3', Block(64, 64, 1, 8))
        self.add_module('b4', Block(64, 128, 2, 8))
        self.add_module('b5', Block(128, 128, 1, 8))
        self.add_module('last', nn.Conv2d(128, 12, 1))
        self.add_module('decode', Decode(num_classes, 16))

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x