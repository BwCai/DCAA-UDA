import os
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models
# import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image
# from visdom import Visdom

_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)

from torch.utils import data
from tqdm import tqdm

from data import create_unimodal_dataset as create_dataset
from models import create_model
from utils.utils import get_logger
from augmentations import get_composed_augmentations
from models.adaptation_model_stage2 import CustomModel, CustomMetrics
from optimizers import get_optimizer
from schedulers import get_scheduler
from metrics import runningScore, averageMeter
from loss import get_loss_function
from utils import sync_batchnorm
from tensorboardX import SummaryWriter
import pdb

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def test(cfg, writer, logger):
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))
    ## create dataset
    default_gpu = cfg['model']['default_gpu']
    device = torch.device("cuda:{}".format(default_gpu) if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(cfg, writer, logger)  #source_train\ target_train\ source_valid\ target_valid + _loader

    model = CustomModel(cfg, writer, logger)
    # source -- ignored
    source_running_metrics_val = runningScore(cfg['data']['target']['n_class'])
    source_val_loss_meter = averageMeter()
    # branches
    modal_num = 3
    running_metrics_val_set = []
    val_loss_meter_set = []
    for _ in range(modal_num + 1):
        running_metrics_val_set.append(
            runningScore(cfg['data']['target']['n_class'])
        )
        val_loss_meter_set.append(
            averageMeter()
        )
        
    time_meter = averageMeter()
    loss_fn = get_loss_function(cfg)
    path = cfg['test']['path']
    checkpoint = torch.load(path)
    model.adaptive_load_nets(model.BaseNet, checkpoint['DeepLab'][0]['model_state'])

    validation(
                model, logger, writer, datasets, device, running_metrics_val_set, val_loss_meter_set, loss_fn,\
                source_val_loss_meter, source_running_metrics_val, iters = model.iter
                )

def validation(model, logger, writer, datasets, device, running_metrics_val, val_loss_meter, loss_fn,\
        source_val_loss_meter, source_running_metrics_val, iters):
    iters = iters
    _k = -1
    model.eval(logger=logger)
    torch.cuda.empty_cache()
    with torch.no_grad():
        validate(
            datasets.target_valid_loader, device, model, running_metrics_val,
            val_loss_meter, loss_fn
            )
        
    attention_running_metrics_val = running_metrics_val[-1] 
    score, class_iou = attention_running_metrics_val.get_scores()
    for k, v in score.items():
        logger.info('attention -- {}: {}'.format(k, v))

    for k, v in class_iou.items():
        logger.info('attention -- {}: {}'.format(k, v))

    attention_running_metrics_val.reset()

    source_val_loss_meter.reset()
    source_running_metrics_val.reset()

    torch.cuda.empty_cache()
    return score["Mean IoU : \t"]

def validate(valid_loader, device, model, running_metrics_val, val_loss_meter, loss_fn, save_fld="result_vis"):
    if not os.path.exists(save_fld):
        os.mkdir(save_fld)
    img_names = []
    img_score_set = []
    img_score_ls = []
    for (images_val, labels_val, filename) in tqdm(valid_loader):

        images_val = images_val.to(device)
        labels_val = labels_val.to(device)
        _, _, feat_cls, outs = model.forward(images_val)

        for _idx, out_b, running_metrics_val_b, val_loss_meter_b in zip(range(len(outs)), outs, running_metrics_val, val_loss_meter):
            if _idx < len(outs)-1:
                continue
            out_b_up = F.interpolate(out_b, size=images_val.size()[2:], mode='bilinear', align_corners=True)
            val_loss = loss_fn(input=out_b_up, target=labels_val)

            pred = out_b_up.max(1)[1].clone()
            gt = labels_val.clone()
            
            img_score = running_metrics_val_b.update(gt, pred)
            img_names.append(os.path.basename(filename[0]))
            img_score_set.append(img_score)
            img_score_ls.append(img_score[0]["Mean IoU : \t"])

            val_loss_meter_b.update(val_loss.item())

            #out_b_up = F.interpolate(out_b, size=(512, 1024), mode='bilinear', align_corners=True)
            #pred_label = out_b_up.argmax(axis=1).squeeze()

            pred = pred.cpu().numpy().squeeze()
            amax_output_vis = np.asarray(pred, dtype=np.uint8)
            amax_output_col = colorize_mask(amax_output_vis)

            img_save_path = '{}/{}'.format(save_fld, os.path.basename(filename[0]).rsplit('.', 1)[0])
            amax_output_col.save(img_save_path + '.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default='configs/test.yml',
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    # path = cfg['training']['save_path']
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    test(cfg, writer, logger)

