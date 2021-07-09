import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.lovasz_losses as L


def set_criterion(method):
    if method == "focal":
        return focal_loss
    elif method == "lovasz":
        return L.lovasz_softmax
    else:  # cross entropy
        return cross_entropy_loss


def cross_entropy_loss(logit, target):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logit, target.long())
    loss /= logit.shape[0]
    return loss


def focal_loss(logit, target):
    gamma = 4
    alpha = 0.9
    criterion = nn.CrossEntropyLoss()

    logpt = -criterion(logit, target.long())
    pt = torch.exp(logpt)
    logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt

    loss /= logit.shape[0]

    return loss


# def lovasz_loss(logit, target):
#     out = F.softmax(logit, dim=1)
#     loss = L.lovasz_softmax(out, target, ignore=0)
#     loss /= loss.shape[0]
#     return loss
