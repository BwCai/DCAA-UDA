import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.aspp import build_aspp
from models.decoder import build_decoder
from models.cam_decoder import build_attention_decoder
from models.backbone import build_backbone

import pdb

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet101', output_stride=16, num_classes=21,
                    bn='bn', freeze_bn=False, modal_num=3):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        self.best_iou = 0
        if bn == 'sync_bn':
            BatchNorm = SynchronizedBatchNorm2d
        elif bn == 'bn':
            BatchNorm = nn.BatchNorm2d
        elif bn == 'gn':
            BatchNorm = nn.GroupNorm
        else:
            raise NotImplementedError('batch norm choice {} is not implemented'.format(bn))

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        # aspp/decoder-branches
        self.modal_num = modal_num
        self.aspps = []
        self.decoders = []
        for item in range(modal_num): 
            self.aspps.append(build_aspp(backbone, output_stride, BatchNorm))
            self.decoders.append(build_decoder(num_classes, backbone, BatchNorm))
        self.aspps = nn.ModuleList(self.aspps)
        self.decoders = nn.ModuleList(self.decoders)

        # attention-branch
        self.attention_decoder = build_attention_decoder(num_classes, modal_num, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def _load_pretrained_model(self):
        if hasattr(self.backbone, '_load_pretrained_model'):
            self.backbone._load_pretrained_model()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)

        cls_feats = []
        outs = []
        for _aspp, _decoder in zip(self.aspps, self.decoders):
            _x = _aspp(x)
            _cls_feat, _out = _decoder(_x, low_level_feat)
            cls_feats.append(_cls_feat)
            outs.append(_out)
        # attention decoder
        att_mask, _att_feat, _att_out = self.attention_decoder(cls_feats, low_level_feat)
        cls_feats.append(_att_feat)
        outs.append(_att_out)
        #assert len(cls_feats) == 4, "cls_feats num:{}".format(len(cls_feats))

        return x, low_level_feat, att_mask, cls_feats, outs

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
            elif isinstance(m, nn.GroupNorm):
                m.eval()
            elif m.__class__.__name__.find('BatchNorm')!= -1:
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        #modules = []
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspps, self.decoders, self.attention_decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params(), 'lr': learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())




