import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.aspp import build_aspp
from models.decoder import build_decoder
import pdb

class AttentionDecoder(nn.Module):
    def __init__(self, num_classes, modal_num, backbone, BatchNorm):
        super(AttentionDecoder, self).__init__()
        backbone = 'resnet'
        if backbone == 'resnet' or backbone == 'drn':
            inplanes = 256 * modal_num
        elif backbone == 'xception':
            inplanes = 128 * modal_num
        elif backbone == 'mobilenet':
            inplanes = 24 * modal_num
        else:
            raise NotImplementedError
        self.modal_num = modal_num

        # attention sequential
        self.att_conv = nn.Sequential(
            nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256) if BatchNorm!=nn.GroupNorm else BatchNorm(16, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, modal_num, kernel_size=1, stride=1, bias=False),
            nn.Softmax(),
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(256 * (modal_num + 1), 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256) if BatchNorm!=nn.GroupNorm else BatchNorm(16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256) if BatchNorm!=nn.GroupNorm else BatchNorm(16, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        self._init_weight()


    def forward(self, x_in, low_level_feat):
        x = x_in.copy()
        _b, _c, _w, _h = x[0].size()
        modal_x = torch.cat(x, dim=1) # B x 2C x W x H
        # attention module
        att_mask = self.att_conv(modal_x) # B x 2 x W x H
        feat_x = x[0] * torch.unsqueeze(att_mask[:, 0, :, :], 1)
        for _i in range(1, self.modal_num):
            feat_x += x[_i] * torch.unsqueeze(att_mask[:, _i, :, :], 1)
        
        x.append(feat_x)
        residual_x = torch.cat(x, dim=1)

        for _j in range(len(self.last_conv)-1):
            residual_x = self.last_conv[_j](residual_x)
        out = self.last_conv[-1](residual_x)
        return att_mask, residual_x, out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_attention_decoder(num_classes, modal_num, backbone, BatchNorm):
    return AttentionDecoder(num_classes, modal_num, backbone, BatchNorm)
