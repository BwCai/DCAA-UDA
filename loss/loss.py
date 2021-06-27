import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb


def cross_entropy2d(input, target, weight=None, size_average=True, is_softmax=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        raise NotImplementedError('sizes of input and label are not consistent')
    if weight is not None:
        weight = torch.Tensor(weight).cuda()

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    if is_softmax:
        loss = F.cross_entropy(
            input, target, weight=weight, size_average=size_average, ignore_index=250
        )
    else:
        EPS = 1e-6
        log_input = (input + EPS).log()
        loss = F.nll_loss(
            log_input, target, weight=weight, size_average=size_average, ignore_index=250
        )
    return loss

def cross_entropy4d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        raise NotImplementedError('sizes of input and label are not consistent')
    if weight is not None:
        weight = torch.Tensor(weight).cuda()

    loss = F.cross_entropy(
        input, target, weight=weight, ignore_index=250, reduction='none'
    )
    return loss

def NLLLoss_2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        raise NotImplementedError('sizes of input and label are not consistent')
    if weight is not None:
        weight = torch.Tensor(weight).cuda()

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.nll_loss(
        torch.log(input), target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss

def multi_scale_cross_entropy2d(
    input, target, weight=None, size_average=True, scale_weight=None
):
    if not isinstance(input, tuple): # when evaluation
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to('cuda' if target.is_cuda else 'cpu')

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input,
                                  target, 
                                  K, 
                                  weight=None, 
                                  size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, 
                                   target, 
                                   K, 
                                   weight=None,
                                   size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input, 
                               target, 
                               weight=weight, 
                               reduce=False,
                               size_average=False, 
                               ignore_index=250)

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)

def balanced_cross_entropy(input, target, beta, size_average=None, ignore_index=250):
    def convert_to_logits(y_pred, EPS=1e-8):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = torch.clamp(y_pred, EPS, 1 - EPS)

        return torch.log(y_pred / (1 - y_pred))

    if input.dim()>2:
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

    target = target.view(-1,1)
    _ids = torch.where(target != ignore_index)[0]
    target = target[_ids]
    input = input[_ids]

    pdb.set_trace()
    t_size, t_c = target.size()
    label = torch.zeros(t_size, 19).cuda()
    id = torch.where(target < 19, target, torch.LongTensor([19]).cuda())
    label = label.scatter_(1, id.long(), 1)

    beta = torch.zeros(input.size(1)).cuda() + beta

    input = convert_to_logits(input)
    pos_weight = beta / (1 - beta)

    input = input[_ids]
    #loss = F.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight, size_average=size_average, reduce=False, reduction=None, ignore_index=250)
    loss = F.binary_cross_entropy_with_logits(input, label, pos_weight=pos_weight, size_average=size_average)

    # or reduce_sum and/or axis=-1
    return loss * (1 - beta)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_label=250):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        _ids = torch.where(target != self.ignore_label)[0] #.cuda()
        target = target[_ids]
        input = input[_ids]

        pdb.set_trace()
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

