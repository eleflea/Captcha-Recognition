import torch
from torch import nn

def smooth_l1_loss(pred, label, beta=1/9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(pred - label)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss.mean(dim=-1)

bce_loss = nn.BCELoss(reduction='none')

def criterion(pred, label):
    mask = label[..., 2:].sum(-1) != 0
    xy_loss = 1 * mask * smooth_l1_loss(pred[..., :2], label[..., :2])
    xy_loss = xy_loss.sum([1, 2]).mean()
    cls_loss = 1 * bce_loss(pred[..., 2:], label[..., 2:])
    cls_loss = cls_loss.sum([1, 2, 3]).mean()
    return xy_loss + cls_loss