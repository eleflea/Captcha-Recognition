import torch
from torch import nn

def build_center_grid(height, width):
    '''
    根据height和width建立一个指定大小的坐标格子
    每个元素是对应格子的中心点坐标(xc, yc)
    '''
    shiftx = torch.arange(0, height, dtype=torch.float32) + 0.5
    shifty = torch.arange(0, width, dtype=torch.float32) + 0.5
    shifty, shiftx = torch.meshgrid(shiftx, shifty)
    xy_grid = torch.stack([shiftx, shifty], dim=-1)
    return xy_grid

class Decode(nn.Module):
    '''
    将CNN的输出转换为真实坐标和概率
    '''
    def __init__(self, num_classes, stride):
        super(Decode, self).__init__()
        self.num_classes = num_classes
        # stride是输入的特征图较原始图像缩小了多少倍
        self.stride = stride
        # 生成5x10的格子
        self.xy_grid = build_center_grid(5, 10)

    def forward(self, conv):
        # 重写排列维度，从(BxCxHxW)->(BxHxWxC)
        conv = conv.permute(0, 2, 3, 1)
        # 将最后一维C分解为2+10
        conv_raw_dxdy, conv_raw_prob = torch.split(conv, [2, self.num_classes], dim=-1)

        # 将xy_grid移入conv所在的设备中
        xy_grid = self.xy_grid.to(conv.device)
        # 计算得到真实坐标
        pred_xy = (xy_grid[None, ...] + torch.sigmoid(conv_raw_dxdy) * 2 - 1) * self.stride
        # 计算得到真实概率
        pred_prob = torch.sigmoid(conv_raw_prob)
        # 将坐标和概率组合回去
        pred = torch.cat((pred_xy, pred_prob), -1)
        return pred

class Block(nn.Module):
    '''
    一个卷积块，由[1x1 conv, ReLU, 3x3 group conv, ReLU, 1x1 conv, BN]组成
    '''
    def __init__(self, w_in, w_out, stride, gw):
        super(Block, self).__init__()
        # 分组数
        g = w_in // gw
        # 卷积
        self.a = nn.Conv2d(w_in, w_in, 1, stride=1, padding=0, bias=False)
        # ReLU
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_in, w_in, 3, stride=stride, padding=1, groups=g, bias=False)
        self.b_relu = nn.ReLU(inplace=True)
        self.c = nn.Conv2d(w_in, w_out, 1, stride=1, padding=0, bias=False)
        # BN
        self.c_bn = nn.BatchNorm2d(w_out)

    def forward(self, x):
        '''
        网络的前向计算过程
        '''
        for layer in self.children():
            x = layer(x)
        return x

class Net(nn.Module):
    '''
    我们定义的网络结构
    '''
    def __init__(self, num_classes):
        super(Net, self).__init__()
        # 第一层是Conv+BN+ReLU
        self.add_module(
            'stem',
            nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
        )
        # 中间层是上面定义的Block
        self.add_module('b1', Block(32, 32, 2, 8))
        self.add_module('b2', Block(32, 64, 2, 8))
        self.add_module('b3', Block(64, 64, 1, 8))
        self.add_module('b4', Block(64, 128, 2, 8))
        self.add_module('b5', Block(128, 128, 1, 8))
        # 最后一层输出通道数为2+10
        self.add_module('last', nn.Conv2d(128, 12, 1))
        # 转换得到预测结果
        self.add_module('decode', Decode(num_classes, 16))

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x