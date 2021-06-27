from models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import torch
import numpy as np
import itertools
import copy

from torch.autograd import Variable
from optimizers import get_optimizer
from schedulers import get_scheduler
from models.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
from models.deeplab_multimodal import DeepLab
from models.decoder import Decoder
from models.aspp import ASPP
from models.discriminator import FCDiscriminator, FCDiscriminator_low, FCDiscriminator_out, FCDiscriminator_class
from loss import get_loss_function
from .utils import freeze_bn, GradReverse, normalisation_pooling
from metrics import runningScore
import pdb

def multimodal_merger(multi_modal_data, is_upsample=False, up_size=None):
    """
    [Func Handler] multimodal_merger:
        @Input Params:
            multi_modal_data: dict.
                examples: {
                    "att_mask": att_mask,
                    "feat_cls": feat_cls,
                    "output": output,
                }
        @Reture:
            merge_out: dict.
                examples: {
                    "feat_cls": feat_cls,
                    "output_comb": output_comb,
                    "output": output,
                }
    """
    att_mask = multi_modal_data['att_mask']
    feat_cls = multi_modal_data['feat_cls']
    # merge class features
    feat_cls_cat = torch.cat(feat_cls, 1) # concat 
    feat_cls_stk = torch.stack(feat_cls, 1) # B x modal_num+1 x 256 x W x H
    assert feat_cls_stk.size(1) == 4 and feat_cls_stk.dim() == 5, "feat_cls_stk.size: {}".format(feat_cls_stk.size())
    # merge output pred
    output = multi_modal_data['output']
    output_comb = 0
    _b, _c, _w, _h = att_mask.size()
    for _i in range(len(output)-1):
        if is_upsample:
            output[_i] = F.interpolate(output[_i], size=up_size, mode='bilinear', align_corners=True)
        output_comb += (output[_i] * att_mask[:,_i,:,:].view(_b, 1, _w, _h))

    if len(output) > 2:
        if is_upsample:
            output[-1] = F.interpolate(output[-1], size=up_size, mode='bilinear', align_corners=True)
        output_comb += output[-1]
        output_comb /= 2.0

    merge_out = {
        'att_mask': att_mask,
        'feat_cls': feat_cls,
        'feat_cls_cat': feat_cls_cat,
        'feat_cls_stk': feat_cls_stk,
        'output_comb': output_comb,
        'output': output,
    }
    return merge_out

def __multimodal_merger_attention(multi_modal_data, is_upsample=False, up_size=None):
    """
    [Func Handler] multimodal_merger:
        @Input Params:
            multi_modal_data: dict.
                examples: {
                    "att_mask": att_mask,
                    "feat_cls": feat_cls,
                    "output": output,
                }
        @Reture:
            merge_out: dict.
                examples: {
                    "feat_cls": feat_cls,
                    "output_comb": output_comb,
                    "output": output,
                }
    """
    att_mask = multi_modal_data['att_mask']
    feat_cls = multi_modal_data['feat_cls']
    # merge class features
    feat_cls_cat = torch.cat(feat_cls, 1) # concat 
    feat_cls_stk = torch.stack(feat_cls, 1) # B x modal_num+1 x 256 x W x H
    assert feat_cls_stk.size(1) == 4 and feat_cls_stk.dim() == 5, "feat_cls_stk.size: {}".format(feat_cls_stk.size())
    # merge output pred
    output = multi_modal_data['output']
    output_comb = 0
    _b, _c, _w, _h = att_mask.size()
    for _i in range(len(output)-1):
        if is_upsample:
            output[_i] = F.interpolate(output[_i], size=up_size, mode='bilinear', align_corners=True)
        output_comb += (output[_i] * att_mask[:,_i,:,:].view(_b, 1, _w, _h))

    if len(output) > 2:
        if is_upsample:
            output[-1] = F.interpolate(output[-1], size=up_size, mode='bilinear', align_corners=True)
        output_comb += output[-1]
        output_comb /= 2.0

    merge_out = {
        'att_mask': att_mask,
        'feat_cls': feat_cls,
        'feat_cls_cat': feat_cls_cat,
        'feat_cls_stk': feat_cls_stk,
        'output_comb': output_comb,
        'output': output,
    }
    return merge_out

def multimodal_merger_voting(multi_modal_data, is_upsample=False, up_size=None):
    """
    [Func Handler] multimodal_merger:
        @Input Params:
            multi_modal_data: dict.
                examples: {
                    "feat_cls": feat_cls,
                    "output": output,
                }
        @Reture:
            merge_out: dict.
                examples: {
                    "feat_cls": feat_cls,
                    "output_comb": output_comb,
                    "output": output,
                }
    """
    feat_cls = multi_modal_data['feat_cls']
    # merge class features
    feat_cls_cat = torch.cat(feat_cls, 1) # concat  
    # merge output pred
    output = multi_modal_data['output']
    output_comb = 0
    for _i in range(len(output)):
        if is_upsample:
            output[_i] = F.interpolate(output[_i], size=up_size, mode='bilinear', align_corners=True)
        output_comb += output[_i]

    merge_out = {
        'feat_cls': feat_cls,
        'feat_cls_cat': feat_cls_cat,
        'output_comb': output_comb,
        'output': output,
    }
    return merge_out

class CustomMetricsMultimodalMerger():
    """
    [Func Handler] objective_vectors_multimodal_merger:
        @Input Params:
            multi_modal_data: dict.
                examples: {
                    "class_threshold_group": [model.class_threshold_group[modal_idx][i], ...]
                    "objective_vectors_group": [model.objective_vectors_group[modal_idx][i], ...],
                }
            cate_idx: int. 0 ~ 18
            modal_ids: list.
                examples: [0, 1] or [0,]
        @Reture:
            merge_out: dict.
                examples: {
                    "class_threshold": class_threshold,
                    "objective_vectors": objective_vectors,
                }
    """

    def __init__(self, modal_num, category_num, model):
        self.modal_num = modal_num
        self.category_num = category_num
        self._model = model

    def initialize_model(model):
        self._model = model

    def merge_class_threshold(self, modal_ids=[]):
        assert self._model is not None, "[ERROR] Deeplab Model not initialize before using!"
        _class_threshold_group = self._model.class_threshold_group[modal_ids]
        return torch.mean(_class_threshold_group, dim=0) # modal_num x 19 --> 19

    def merge_clu_threshold(self, clu_threshold, modal_ids=[]):
        _clu_threshold_group = clu_threshold[modal_ids]
        return torch.mean(_clu_threshold_group, dim=0)

    def merge_objective_vectors(self, modal_ids=[]):
        assert self._model is not None, "[ERROR] Deeplab Model not initialize before using!"
        _modal_num, _cate_num, _feat_dim = self._model.objective_vectors_group.size()
        _objective_vectors = self._model.objective_vectors_group[modal_ids]
        # modal_num x 19 x 256 --> 19 x modal_num x 256 --> 19 x (modal_num x 256)
        assert _objective_vectors.dim() == 3, "objective_vector dimension != 3"
        _objective_vectors = _objective_vectors.permute(1, 0, 2).contiguous()

        return _objective_vectors
        #return _objective_vectors.view(_cate_num, -1)

class CustomMetrics():
    def __init__(self, numbers=19, modal_num=3, model=None):
        self.class_numbers = numbers
        self.classes_recall_thr = np.zeros([19, 3])
        self.classes_recall_thr_num = np.zeros([19])
        self.classes_recall_clu = np.zeros([19, 3])
        self.classes_recall_clu_num = np.zeros([19])
        self.running_metrics_val_threshold = runningScore(self.class_numbers)
        self.running_metrics_val_clusters = runningScore(self.class_numbers)
        self.clu_threshold = torch.full((modal_num + 1, 19), 3.0).cuda()
        self.multimodal_merger = CustomMetricsMultimodalMerger(
            modal_num=modal_num + 1, category_num=numbers, model=model
        )
        #self.kl_distance = nn.KLDivLoss(reduction='none')
    
    def update(self, feat_cls, outputs, labels, modal_ids=[0,], att_mask=None):
        '''calculate accuracy. caring about recall but not IoU'''
        if att_mask is not None:
            assert len(modal_ids) == 4, "modal_ids: {}".format(modal_ids)
            _outputs = outputs
            _outputs = []
            for out_b in outputs:
                _outputs.append(F.softmax(out_b, dim=1))

            multi_modal_data = {
                "att_mask": att_mask,
                "feat_cls": feat_cls,
                "output": _outputs,
            }
            merger_out = multimodal_merger(multi_modal_data, is_upsample=False, up_size=None)
            feat_cls = merger_out['feat_cls_stk']

            outputs = merger_out['output_comb']
            outputs_threshold = merger_out['output_comb'].clone()
        else:
            outputs_threshold = outputs.clone()
            outputs_threshold = F.softmax(outputs_threshold, dim=1)

        batch, width, height = labels.shape
        labels = labels.reshape([batch, 1, width, height]).float()
        labels = F.interpolate(labels, size=feat_cls[0].size()[-2:], mode='nearest')

        self.running_metrics_val_threshold.update(labels, outputs_threshold.argmax(1))
        _class_threshold_set = self.multimodal_merger.merge_class_threshold(modal_ids=modal_ids)
        for i in range(19):
            outputs_threshold[:, i, :, :] = torch.where(outputs_threshold[:, i, :, :] > _class_threshold_set[i], torch.Tensor([1]).cuda(), torch.Tensor([0]).cuda())
        _batch, _channel, _w, _h = outputs_threshold.shape
        _tmp = torch.full([_batch, 1, _w, _h], 0.2,).cuda()
        _tmp = torch.cat((outputs_threshold, _tmp), 1)
        threshold_arg = _tmp.argmax(1, keepdim=True)
        threshold_arg[threshold_arg == 19] = 250 #ignore index

        truth, pred_all, truth_all = self.calc_recall(labels.cpu().int().numpy(), threshold_arg.cpu().int().numpy())
        self.classes_recall_thr[:, 0] += truth
        self.classes_recall_thr[:, 2] += pred_all
        self.classes_recall_thr[:, 1] += truth_all

        # feature cluster
        outputs_cluster = outputs.clone()
        _objective_vectors_set = self.multimodal_merger.merge_objective_vectors(modal_ids=modal_ids) # modal_num+1 x 19 x 256

        _modal_num = len(modal_ids)
        if _modal_num == 1:
            _b, _c, _w, _h = feat_cls.shape
        else:
            _b, _m, _c, _w, _h = feat_cls.shape

        tmp_mask = torch.full([_b, 1, _w, _h], 1.0,).cuda()

        if _modal_num > 1:
            tmp_mask = torch.cat((att_mask, tmp_mask), 1)
            for i in range(19):
                outputs_cluster[:, i, :, :] = torch.sum(
                    torch.norm( _objective_vectors_set[i].reshape(_modal_num,-1,1,1).expand(_modal_num,-1,128,224) - feat_cls, 2, dim=2,) * tmp_mask, 
                    dim=1,
                ) / 2.0
        else:
            for i in range(19):
                outputs_cluster[:, i, :, :] = torch.norm(_objective_vectors_set[i].reshape(-1,1,1).expand(-1,128,224) - feat_cls, 2, dim=1,)
        outputs_cluster_min, outputs_cluster_arg = outputs_cluster.min(dim=1, keepdim=True)
        outputs_cluster_second = outputs_cluster.scatter_(1, outputs_cluster_arg, 100)
        if torch.unique(outputs_cluster_second.argmax(1) - outputs_cluster_arg.squeeze()).squeeze().item() != 0:
            raise NotImplementedError('wrong when computing L2 norm!!')
        outputs_cluster_secondmin, outputs_cluster_secondarg = outputs_cluster_second.min(dim=1, keepdim=True)
        
        self.running_metrics_val_clusters.update(labels, outputs_cluster_arg)
        
        tmp_arg = outputs_cluster_arg.clone()
        _clu_thresholds = self.multimodal_merger.merge_clu_threshold(self.clu_threshold, modal_ids=modal_ids)

        outputs_cluster_arg[(outputs_cluster_secondmin - outputs_cluster_min) < _clu_thresholds[tmp_arg]] = 250
        truth, pred_all, truth_all = self.calc_recall(labels.cpu().int().numpy(), outputs_cluster_arg.cpu().int().numpy())
        self.classes_recall_clu[:, 0] += truth
        self.classes_recall_clu[:, 2] += pred_all
        self.classes_recall_clu[:, 1] += truth_all
        return threshold_arg, outputs_cluster_arg

    def calc_recall(self, gt, argmax):
        truth = np.zeros([self.class_numbers])
        pred_all = np.zeros([self.class_numbers])
        truth_all = np.zeros([self.class_numbers])
        for i in range(self.class_numbers):
            truth[i] = (gt == i)[argmax == i].sum()
            pred_all[i] = (argmax == i).sum()
            truth_all[i] = (gt == i).sum()
        pass
        return truth, pred_all, truth_all
    
    def calc_mean_Clu_recall(self, ):
        return np.mean(self.classes_recall_clu[:, 0] / self.classes_recall_clu[:, 1])
    
    def calc_mean_Thr_recall(self, ):
        return np.mean(self.classes_recall_thr[:, 0] / self.classes_recall_thr[:, 1])

    def reset(self, ):
        self.running_metrics_val_clusters.reset()
        self.running_metrics_val_threshold.reset()
        self.classes_recall_clu = np.zeros([19, 3])
        self.classes_recall_thr = np.zeros([19, 3])

class CustomModel():
    def __init__(self, cfg, writer, logger, use_pseudo_label=False, modal_num=3, multimodal_merger=multimodal_merger):
        self.cfg = cfg
        self.writer = writer
        self.class_numbers = 19
        self.logger = logger
        cfg_model = cfg['model']
        self.cfg_model = cfg_model
        self.best_iou = -100
        self.iter = 0
        self.nets = []
        self.split_gpu = 0
        self.default_gpu = cfg['model']['default_gpu']
        self.PredNet_Dir = None
        self.valid_classes = cfg['training']['valid_classes']
        self.G_train = True
        self.cls_feature_weight = cfg['training']['cls_feature_weight']
        self.use_pseudo_label = use_pseudo_label
        self.modal_num = modal_num

        # cluster vectors & cuda initialization
        self.objective_vectors_group = torch.zeros(self.modal_num + 1, 19, 256).cuda()
        self.objective_vectors_num_group = torch.zeros(self.modal_num + 1, 19).cuda()
        self.objective_vectors_dis_group = torch.zeros(self.modal_num + 1, 19, 19).cuda()
        self.class_threshold_group = torch.full([self.modal_num + 1, 19], 0.6).cuda()

        self.disc_T = torch.FloatTensor([0.0]).cuda()

        #self.metrics = CustomMetrics(self.class_numbers)
        self.metrics = CustomMetrics(self.class_numbers, modal_num=self.modal_num, model=self)

        # multimodal / multi-branch merger
        self.multimodal_merger = multimodal_merger

        bn = cfg_model['bn']
        if bn == 'sync_bn':
            BatchNorm = SynchronizedBatchNorm2d
        elif bn == 'bn':
            BatchNorm = nn.BatchNorm2d
        elif bn == 'gn':
            BatchNorm = nn.GroupNorm
        else:
            raise NotImplementedError('batch norm choice {} is not implemented'.format(bn))

        if True:
            self.PredNet = DeepLab(
                    num_classes=19,
                    backbone=cfg_model['basenet']['version'],
                    output_stride=16,
                    bn=cfg_model['bn'],
                    freeze_bn=True,
                    modal_num=self.modal_num
                    ).cuda()
            self.load_PredNet(cfg, writer, logger, dir=None, net=self.PredNet)
            self.PredNet_DP = self.init_device(self.PredNet, gpu_id=self.default_gpu, whether_DP=True) 
            self.PredNet.eval()
            self.PredNet_num = 0

            self.PredDnet = FCDiscriminator(inplanes=19)
            self.load_PredDnet(cfg, writer, logger, dir=None, net=self.PredDnet)
            self.PredDnet_DP = self.init_device(self.PredDnet, gpu_id=self.default_gpu, whether_DP=True)
            self.PredDnet.eval()

        self.BaseNet = DeepLab(
                            num_classes=19,
                            backbone=cfg_model['basenet']['version'],
                            output_stride=16,
                            bn=cfg_model['bn'],
                            freeze_bn=True, 
                            modal_num=self.modal_num
                            )

        logger.info('the backbone is {}'.format(cfg_model['basenet']['version']))

        self.BaseNet_DP = self.init_device(self.BaseNet, gpu_id=self.default_gpu, whether_DP=True)
        self.nets.extend([self.BaseNet])
        self.nets_DP = [self.BaseNet_DP]

        # Discriminator
        self.SOURCE_LABEL = 0
        self.TARGET_LABEL = 1
        self.DNets = []
        self.DNets_DP = []
        for _ in range(self.modal_num+1):
            _net_d = FCDiscriminator(inplanes=19)
            self.DNets.append(_net_d)
            _net_d_DP = self.init_device(_net_d, gpu_id=self.default_gpu, whether_DP=True)
            self.DNets_DP.append(_net_d_DP)

        self.nets.extend(self.DNets)
        self.nets_DP.extend(self.DNets_DP)

        self.optimizers = []
        self.schedulers = []        

        optimizer_cls = torch.optim.SGD
        optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() 
                            if k != 'name'}

        optimizer_cls_D = torch.optim.Adam
        optimizer_params_D = {k:v for k, v in cfg['training']['optimizer_D'].items() 
                            if k != 'name'}

        if False:
            self.BaseOpti = optimizer_cls(self.BaseNet.parameters(), **optimizer_params)
        else:
            self.BaseOpti = optimizer_cls(self.BaseNet.optim_parameters(cfg['training']['optimizer']['lr']), **optimizer_params)

        self.optimizers.extend([self.BaseOpti])

        self.DiscOptis = []
        for _d_net in self.DNets: 
            self.DiscOptis.append(
                optimizer_cls_D(_d_net.parameters(), **optimizer_params_D)
            )
        self.optimizers.extend(self.DiscOptis)

        self.schedulers = []        

        if False:
            self.BaseSchedule = get_scheduler(self.BaseOpti, cfg['training']['lr_schedule'])
            self.schedulers.extend([self.BaseSchedule])
        else:
            """BaseSchedule detail see FUNC: scheduler_step()"""
            self.learning_rate = cfg['training']['optimizer']['lr']
            self.gamma = cfg['training']['lr_schedule']['gamma']
            self.num_steps = cfg['training']['lr_schedule']['max_iter']
            self._BaseSchedule_nouse = get_scheduler(self.BaseOpti, cfg['training']['lr_schedule'])
            self.schedulers.extend([self._BaseSchedule_nouse])

        self.DiscSchedules = []
        for _disc_opt in self.DiscOptis:
            self.DiscSchedules.append(
                get_scheduler(_disc_opt, cfg['training']['lr_schedule'])
            )
        self.schedulers.extend(self.DiscSchedules)
        self.setup(cfg, writer, logger)

        self.adv_source_label = 0
        self.adv_target_label = 1
        self.bceloss = nn.BCEWithLogitsLoss(reduce=False)
        self.loss_fn = get_loss_function(cfg)
        pseudo_cfg = copy.deepcopy(cfg)
        pseudo_cfg['training']['loss']['name'] = 'cross_entropy4d'
        self.pseudo_loss_fn = get_loss_function(pseudo_cfg)
        self.mseloss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.smoothloss = nn.SmoothL1Loss()
        self.triplet_loss = nn.TripletMarginLoss()
        self.kl_distance = nn.KLDivLoss(reduction='none')

    def create_PredNet(self,):
        ss = DeepLab(
                num_classes=19,
                backbone=self.cfg_model['basenet']['version'],
                output_stride=16,
                bn=self.cfg_model['bn'],
                freeze_bn=True,
                modal_num=self.modal_num,
                ).cuda()
        ss.eval()
        return ss

    def setup(self, cfg, writer, logger):
        '''
        set optimizer and load pretrained model
        '''
        for net in self.nets:
            # name = net.__class__.__name__
            self.init_weights(cfg['model']['init'], logger, net)
            print("Initializition completed")
            if hasattr(net, '_load_pretrained_model') and cfg['model']['pretrained']:
                print("loading pretrained model for {}".format(net.__class__.__name__))
                net._load_pretrained_model()
        '''load pretrained model
        '''
        if cfg['training']['resume_flag']:
            self.load_nets(cfg, writer, logger)
        pass

    def lr_poly(self):
        return self.learning_rate * ((1 - float(self.iter) / self.num_steps) ** (self.gamma))

    def adjust_basenet_learning_rate(self):
        lr = self.lr_poly()
        self.BaseOpti.param_groups[0]['lr'] = lr
        if len(self.BaseOpti.param_groups) > 1:
            self.BaseOpti.param_groups[1]['lr'] = lr * 10

    def forward(self, input):
        feat, feat_low, att_mask, feat_cls, output = self.BaseNet_DP(input)
        return feat, feat_low, feat_cls, output

    def forward_Up(self, input):
        feat, feat_low, feat_cls, outputs = self.forward(input)
        merge_out = self.multimodal_merger(
            {
                'feat_cls': feat_cls,
                'output': output,
            },
            is_upsample=True,
            size=input.size()[2:],
        )
        return feat, feat_low, merge_out['feat_cls'], merge_out['output_comb']

    def PredNet_Forward(self, input):
        with torch.no_grad():
            _, _, att_mask, feat_cls, output_result = self.PredNet_DP(input)
        return _, _, att_mask, feat_cls, output_result

    def calculate_mean_vector(self, feat_cls, outputs, labels, ):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        labels_expanded = self.process_label(labels)
        outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                scale = torch.sum(outputs_pred[n][t]) / labels.shape[2] / labels.shape[3] * 2
                s = normalisation_pooling()(s, scale)
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def step(self, source_x, source_label, source_modal_ids, target_x, target_label, target_modal_ids, use_pseudo_loss=False):
        assert len(source_modal_ids) == source_x.size(0), "modal_ids' batchsize != source_x's batchsize"
        _, _, source_feat_cls, source_output = self.forward(input=source_x) 
        """source_output: [B x 19 x W x H, ...]
        select modal-branch output in each batchsize
        Specific-modal output
        """
        source_output_modal_k = torch.stack(
            [
                source_output[_modal_i][_batch_i]
                for _batch_i, _modal_i in enumerate(source_modal_ids)
            ], 
            dim=0,
        )
        # attention output & specific-modal output
        source_output_comb = torch.cat([source_output_modal_k, source_output[-1]], dim=0)

        source_label_comb = torch.cat([source_label, source_label.clone()], dim=0)

        source_outputUp = F.interpolate(source_output_comb, size=source_x.size()[-2:], mode='bilinear', align_corners=True)

        loss_GTA = self.loss_fn(input=source_outputUp, target=source_label_comb)

        self.PredNet.eval()
        with torch.no_grad():
            _, _, att_mask, feat_cls, output = self.PredNet_Forward(target_x)

            threshold_args_comb, cluster_args_comb = self.metrics.update(feat_cls, output, target_label, modal_ids=[_i for _i in range(self.modal_num+1)], att_mask=att_mask)

            """ Discriminator-guided easy/hard training """
            target_label_size = target_label.size()
            t_out = output[-1]
            _t_out = F.interpolate(t_out.detach(), size=(target_label_size[1]*4, target_label_size[2]*4), mode='bilinear', align_corners=True)
            _t_D_out = self.PredDnet_DP(F.softmax(_t_out))
            _t_D_out_prob = F.sigmoid(_t_D_out)

            disc_easy_weight = torch.where(_t_D_out_prob > self.disc_T, _t_D_out_prob, torch.FloatTensor([0.0]).cuda())
            disc_easy_weight = torch.where(threshold_args_comb != 250, disc_easy_weight, torch.FloatTensor([0.0]).cuda()).squeeze(1)

            disc_hard_mask = torch.where(_t_D_out_prob < self.disc_T, torch.Tensor([1]).cuda(), torch.Tensor([0]).cuda())
            disc_hard_mask = torch.where(threshold_args_comb == 250, torch.Tensor([1]).cuda(), disc_hard_mask)


        loss_L2_source_cls = torch.Tensor([0]).cuda(self.split_gpu)
        loss_L2_target_cls = torch.Tensor([0]).cuda(self.split_gpu)
        _, _, target_feat_cls, target_output = self.forward(target_x)

        if self.cfg['training']['loss_L2_cls']:     # distance loss
            _batch, _w, _h = source_label.shape
            source_label_downsampled = source_label.reshape([_batch,1,_w, _h]).float()
            source_label_downsampled = F.interpolate(source_label_downsampled.float(), size=source_feat_cls[0].size()[-2:], mode='nearest')   #or F.softmax(input=source_output, dim=1)

            loss_L2_source_cls = torch.Tensor([0]).cuda()
            loss_L2_target_cls = torch.Tensor([0]).cuda()
            for _modal_i, _source_feat_i, _source_out_i, _target_feat_i, _target_out_i in zip(range(self.modal_num + 1), source_feat_cls, source_output, target_feat_cls, target_output):
                if _modal_i < 2:
                    continue
                source_vectors, source_ids = self.calculate_mean_vector(_source_feat_i, _source_out_i, source_label_downsampled)
                loss_L2_source_cls += self.class_vectors_alignment(source_ids, source_vectors, modal_ids=[_modal_i,])

                target_vectors, target_ids = self.calculate_mean_vector(_target_feat_i, _target_out_i, cluster_args_comb.float())
                loss_L2_target_cls += self.class_vectors_alignment(target_ids, target_vectors, modal_ids=[_modal_i,])

        loss_L2_cls = self.cls_feature_weight * (loss_L2_source_cls + loss_L2_target_cls)
        if loss_L2_cls.item() > 1.0:
            loss_L2_cls = loss_L2_cls / 10.0
        
        if loss_L2_cls.item() > 0.5:
            loss_L2_cls = loss_L2_cls / 3.0

        target_label_size = target_label.size()

        loss = torch.Tensor([0]).cuda()
        batch, _, w, h = threshold_args_comb.shape
        _cluster_args_comb = cluster_args_comb.reshape([batch, w, h])
        _threshold_args_comb = threshold_args_comb.reshape([batch, w, h])
        _target_output = target_output[-1]

        _loss_CTS = self.pseudo_loss_fn(input=_target_output, target=_threshold_args_comb)  # CAG-based and probability-based PLA
        loss_CTS = torch.sum(_loss_CTS * disc_easy_weight) / (1 + (disc_easy_weight > 0).sum())

        if self.G_train and self.cfg['training']['loss_pseudo_label']:
            loss = loss + loss_CTS
        if self.G_train and self.cfg['training']['loss_source_seg']:
            loss = loss + loss_GTA
        if self.cfg['training']['loss_L2_cls']:
            loss = loss + torch.sum(loss_L2_cls)

        # adversarial loss
        # -----------------------------
        """Generator (segmentation)"""
        # -----------------------------

        # On Source Domain 
        loss_adv = torch.Tensor([0]).cuda()
        _batch_size = 0

        source_modal_ids_tensor = torch.Tensor(source_modal_ids).cuda()
        target_modal_ids_tensor = torch.Tensor(target_modal_ids).cuda()
        for t_out, _d_net_DP, _d_net, modal_idx in zip(target_output, self.DNets_DP, self.DNets, range(len(target_output))):
            # set grad false
            self.set_requires_grad(self.logger, _d_net, requires_grad = False)
            t_D_out = _d_net_DP(F.softmax(t_out))
            _disc_hard_mask = F.interpolate(disc_hard_mask, size=(t_D_out.size(2), t_D_out.size(3)), mode='nearest')
            #source_modal_ids
            loss_temp = torch.sum(self.bceloss(
                t_D_out,
                torch.FloatTensor(t_D_out.data.size()).fill_(1.0).cuda()
            ) * _disc_hard_mask, [1,2,3]) / (torch.sum(disc_hard_mask, [1,2,3]) + 1)

            if modal_idx >= self.modal_num:
                loss_adv += torch.mean(loss_temp)
            elif torch.mean(torch.as_tensor((modal_idx==target_modal_ids_tensor), dtype=torch.float32)) == 0:
                loss_adv += 0.0
            else:
                loss_adv += torch.mean(torch.masked_select(loss_temp, target_modal_ids_tensor==modal_idx))

            _batch_size += t_out.size(0)

        loss_adv *= self.cfg['training']['loss_adv_lambda']

        loss_G = torch.Tensor([0]).cuda()
        loss_G = loss_G + loss_adv
        loss = loss + loss_G
        if loss.item() != 0:
            loss.backward()

        self.BaseOpti.step()
        self.BaseOpti.zero_grad()

        # -----------------------------
        """Discriminator """
        # -----------------------------            
        _batch_size = 0
        loss_D_comb = torch.Tensor([0]).cuda()
        source_label_size = source_label.size()
        for s_out, t_out, _d_net_DP, _d_net, _disc_opt, modal_idx in zip(source_output, target_output, self.DNets_DP, self.DNets, self.DiscOptis, range(len(source_output))):
            self.set_requires_grad(self.logger, _d_net, requires_grad = True)
            
            _batch_size = 0
            loss_D = torch.Tensor([0]).cuda()
            # source domain
            s_D_out = _d_net_DP(F.softmax(s_out.detach()))

            loss_temp_s = torch.mean(self.bceloss(
                s_D_out,
                torch.FloatTensor(s_D_out.data.size()).fill_(1.0).cuda()
            ), [1,2,3])

            if modal_idx >= self.modal_num:
                loss_D += torch.mean(loss_temp_s)
            elif torch.mean(torch.as_tensor((modal_idx==source_modal_ids_tensor), dtype=torch.float32)) == 0:
                loss_D += 0.0
            else:
                loss_D += torch.mean(torch.masked_select(loss_temp_s, source_modal_ids_tensor==modal_idx))

            # target domain
            _batch_size += (s_out.size(0) + t_out.size(0))
            t_D_out = _d_net_DP(F.softmax(t_out.detach()))
            loss_temp_t = torch.mean(self.bceloss(
                t_D_out,
                torch.FloatTensor(t_D_out.data.size()).fill_(0.0).cuda()
            ), [1,2,3])

            if modal_idx >= self.modal_num:
                loss_D += torch.mean(loss_temp_t)
            elif torch.mean(torch.as_tensor((modal_idx==target_modal_ids_tensor), dtype=torch.float32)) == 0:
                loss_D += 0.0
            else:
                loss_D += torch.mean(torch.masked_select(loss_temp_t, target_modal_ids_tensor==modal_idx))

            loss_D *= self.cfg['training']['loss_adv_lambda']*0.5
            if loss_D.item() != 0:
                loss_D.backward()

            _disc_opt.step()
            _disc_opt.zero_grad()

            loss_D_comb += loss_D

        return loss, loss_adv, loss_D_comb


    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, 20, w, h).cuda()
        id = torch.where(label < 19, label, torch.Tensor([19]).cuda())
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1

    def class_vectors_alignment(self, ids, vectors, modal_ids=[0,]):
        loss = torch.Tensor([0]).cuda()

        """construct category objective vectors"""
        # objective_vectors_group 2 x 19 x 256 --> 19 x 512
        _objective_vectors_set = self.metrics.multimodal_merger.merge_objective_vectors(modal_ids=modal_ids)

        for i in range(len(ids)):
            if ids[i] not in self.valid_classes:
                continue
            new_loss = self.smoothloss(vectors[i].squeeze().cuda(), _objective_vectors_set[ids[i]])
            while (new_loss.item() > 5):
                new_loss = new_loss / 10
            loss = loss + new_loss
        loss = loss / len(ids) * 10
        return loss

    def freeze_bn_apply(self):
        for net in self.nets:
            net.apply(freeze_bn)
        for net in self.nets_DP:
            net.apply(freeze_bn)

    def scheduler_step(self):
        if self.use_pseudo_label:
            for scheduler in self.schedulers:
                scheduler.step()
        else:
            """skipped _BaseScheduler_nouse"""
            for scheduler in self.schedulers[1:]:
                scheduler.step()
            self.adjust_basenet_learning_rate()
    
    def optimizer_zerograd(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    
    def optimizer_step(self):
        for opt in self.optimizers:
            opt.step()

    def init_device(self, net, gpu_id=None, whether_DP=False):
        gpu_id = gpu_id or self.default_gpu
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
        net = net.to(device)

        if whether_DP:
            net = DataParallelWithCallback(net, device_ids=range(torch.cuda.device_count()))
        return net
    
    def eval(self, net=None, logger=None):
        """Make specific models eval mode during test time"""
        if net == None:
            for net in self.nets:
                net.eval()
            for net in self.nets_DP:
                net.eval()
            if logger!=None:    
                logger.info("Successfully set the model eval mode") 
        else:
            net.eval()
            if logger!=None:    
                logger("Successfully set {} eval mode".format(net.__class__.__name__))
        return

    def train(self, net=None, logger=None):
        if net==None:
            for net in self.nets:
                net.train()
            for net in self.nets_DP:
                net.train()
        else:
            net.train()
        return

    def set_requires_grad(self, logger, net, requires_grad = False):
        """Set requires_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            net (BaseModel)       -- the network which will be operated on
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        for parameter in net.parameters():
            parameter.requires_grad = requires_grad
        
    def set_requires_grad_layer(self, logger, net, layer_type='batchnorm', requires_grad=False):  
        '''    set specific type of layers whether needing grad
        '''

        # print('Warning: all the BatchNorm params are fixed!')
        # logger.info('Warning: all the BatchNorm params are fixed!')
        for net in self.nets:
            for _i in net.modules():
                if _i.__class__.__name__.lower().find(layer_type.lower()) != -1:
                    _i.weight.requires_grad = requires_grad
        return

    def init_weights(self, cfg, logger, net, init_type='normal', init_gain=0.02):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        init_type = cfg.get('init_type', init_type)
        init_gain = cfg.get('init_gain', init_gain)
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, SynchronizedBatchNorm2d) or classname.find('BatchNorm2d') != -1 \
                or isinstance(m, nn.GroupNorm):
                # or isinstance(m, InPlaceABN) or isinstance(m, InPlaceABNSync):
                m.weight.data.fill_(1)
                m.bias.data.zero_() # BatchNorm Layer's weight is not a matrix; only normal distribution applies.


        print('initialize {} with {}'.format(init_type, net.__class__.__name__))
        logger.info('initialize {} with {}'.format(init_type, net.__class__.__name__))
        net.apply(init_func)  # apply the initialization function <init_func>
        pass

    def adaptive_load_nets(self, net, model_weight):
        model_dict = net.state_dict()
        pretrained_dict = {k : v for k, v in model_weight.items() if k in model_dict}
        
        print("[INFO] Pretrained dict:", pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    def load_nets(self, cfg, writer, logger):    # load pretrained weights on the net
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            _k = -1
            net_state_no = {}
            for net in self.nets:
                name = net.__class__.__name__
                if name not in net_state_no:
                    net_state_no[name] = 0
                else:
                    net_state_no[name] += 1
                _k += 1
                if checkpoint.get(name) == None:
                    continue
                if name.find('FCDiscriminator') != -1 and cfg['training']['gan_resume'] == False:
                    continue
                #self.adaptive_load_nets(net, checkpoint[name]["model_state"])
                if isinstance(checkpoint[name], list):
                    self.adaptive_load_nets(net, checkpoint[name][net_state_no[name]]["model_state"])
                else:
                    print("*****************************************")
                    print("[WARNING] Using depreciated load version! Model {}".format(name))
                    print("*****************************************")
                    self.adaptive_load_nets(net, checkpoint[name]["model_state"])
                if cfg['training']['optimizer_resume']:
                    if isinstance(checkpoint[name], list):
                        self.adaptive_load_nets(self.optimizers[_k], checkpoint[name][net_state_no[name]]["optimizer_state"])
                        self.adaptive_load_nets(self.schedulers[_k], checkpoint[name][net_state_no[name]]["scheduler_state"])
                    else:
                        self.adaptive_load_nets(self.optimizers[_k], checkpoint[name]["optimizer_state"])
                        self.adaptive_load_nets(self.schedulers[_k], checkpoint[name]["scheduler_state"])
            self.iter = checkpoint["iter"] if 'iter' in checkpoint else 0
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], self.iter
                )
            )
        else:
            raise Exception("No checkpoint found at '{}'".format(cfg['training']['resume']))


    def load_PredNet(self, cfg, writer, logger, dir=None, net=None):    # load pretrained weights on the net
        dir = dir or cfg['training']['Pred_resume']
        best_iou = 0
        if os.path.isfile(dir):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(dir)
            )
            checkpoint = torch.load(dir)
            name = net.__class__.__name__
            if checkpoint.get(name) == None:
                return
            if name.find('FCDiscriminator') != -1 and cfg['training']['gan_resume'] == False:
                return
            if isinstance(checkpoint[name], list):
                self.adaptive_load_nets(net, checkpoint[name][0]["model_state"])
            else:
                self.adaptive_load_nets(net, checkpoint[name]["model_state"])
            if 'iter' in checkpoint:
                checkpoint_iter = checkpoint["iter"]
            else:
                checkpoint_iter = 0
            if 'best_iou' in checkpoint:
                best_iou = checkpoint['best_iou']
            else:
                best_iou = 0
            logger.info(
                "Loaded checkpoint '{}' (iter {}) (best iou {}) for PredNet".format(
                    dir, checkpoint_iter, best_iou
                )
            )
        else:
            raise Exception("No checkpoint found at '{}'".format(dir))
        if hasattr(net, 'best_iou'):
            pass
        return best_iou

    def load_PredDnet(self, cfg, writer, logger, dir=None, net=None):    # load pretrained weights on the net
        dir = dir or cfg['training']['Pred_resume']
        best_iou = 0
        if os.path.isfile(dir):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(dir)
            )
            checkpoint = torch.load(dir)
            name = net.__class__.__name__
            if checkpoint.get(name) == None:
                return
            if name.find('FCDiscriminator') != -1 and cfg['training']['gan_resume'] == False:
                return
            if isinstance(checkpoint[name], list):
                self.adaptive_load_nets(net, checkpoint[name][-1]["model_state"]) # attention-branch discriminator
            else:
                print("[WARNING] load discriminator maybe error!")
                self.adaptive_load_nets(net, checkpoint[name]["model_state"])
            print("[INFO] {}: {}".format(name, net))
            iter = checkpoint["iter"]
            logger.info(
                "Loaded checkpoint '{}' (iter {}) for PredNet".format(
                    dir, checkpoint["iter"]
                )
            )
        else:
            raise Exception("No checkpoint found at '{}'".format(dir))
        return best_iou


    def set_optimizer(self, optimizer):  #set optimizer to all nets
        pass

    def reset_objective_SingleVector(self,):
        self.objective_vectors_group = torch.zeros(self.modal_num + 1, 19, 256).cuda()
        self.objective_vectors_num_group = torch.zeros(self.modal_num + 1, 19).cuda()
        self.objective_vectors_dis_group = torch.zeros(self.modal_num + 1, 19, 19).cuda()

    def update_objective_SingleVector(self, vectors, vectors_num, name='moving_average'):
        if torch.sum(vectors) == 0:
            return
        """
        if self.objective_vectors_num_group[modal_idx][id] < 100:
            name = 'mean'
        """
        if name == 'moving_average':
            self.objective_vectors_group = self.objective_vectors_group * 0.9999 + 0.0001 * vectors
            self.objective_vectors_num_group += vectors_num
            self.objective_vectors_num_group = min(self.objective_vectors_num_group, 3000)
        elif name == 'mean':
            self.objective_vectors_group = self.objective_vectors_group * self.objective_vectors_num_group.view(-1, 19, 1).expand(self.modal_num+1, 19, 256) + vectors
            self.objective_vectors_num_group = self.objective_vectors_num_group + vectors_num
            _objective_vectors_num_group = self.objective_vectors_num_group.clone()
            _ids = torch.where(_objective_vectors_num_group == 0)
            _objective_vectors_num_group[_ids] = 1.0
            self.objective_vectors_group = self.objective_vectors_group / _objective_vectors_num_group.view(-1, 19, 1).expand(self.modal_num+1, 19, 256)
            self.objective_vectors_num_group = torch.min(self.objective_vectors_num_group, torch.Tensor([3000]).cuda())
        else:
            raise NotImplementedError('no such updating way of objective vectors {}'.format(name))


def grad_reverse(x):
    return GradReverse()(x)

