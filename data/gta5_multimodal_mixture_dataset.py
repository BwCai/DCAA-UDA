import os
import sys
import torch
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import matplotlib.image  as imgs
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import scipy.io as io
from tqdm import tqdm
from scipy import stats

from torch.utils import data

from .utils import recursive_glob
from augmentations import get_composed_augmentations
from data import BaseDataset
import pdb

class GTA5_Multimodal_Mixture_loader(BaseDataset):
    """
    GTA5    synthetic dataset
    for domain adaptation to Cityscapes
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))
    def __init__(
        self, 
        cfg,
        writer,
        logger,
        augmentations=None,
    ):
        self.cfg = cfg
        self.root_Ds = cfg['rootpath']
        self.split = cfg['split']
        self.is_transform = cfg.get('is_transform', True)
        self.augmentations = augmentations
        self.img_norm = cfg.get('img_norm', True)
        self.n_classes = 19
        self.img_size = (
            cfg['img_cols'], cfg['img_rows']
        )
        self.paired_files = {}

        self.mean = [0.0, 0.0, 0.0] #TODO:  calculating the mean value of rgb channels on GTA5
        # Domain base
        self.image_base_paths = []
        self.label_base_paths = []
        for _root_i in self.root_Ds:
            self.image_base_paths.append(
                os.path.join(_root_i, 'images')
            )
            self.label_base_paths.append(
                os.path.join(_root_i, 'labels')
            )
        self.distribute = np.zeros(self.n_classes, dtype=float)
        # splits = io.loadmat(os.path.join(self.root, 'split.mat'))
        self.ids = []
        for _i, _label_base_p in enumerate(self.label_base_paths):
            _name_set = recursive_glob(rootdir=_label_base_p, suffix=".png")
            _name_set_dict = [
                {
                    "filename": _name_i, 
                    "image_path": os.path.join(self.image_base_paths[_i], _name_i.split("/")[-1]),
                    "gt_path": _name_i, 
                } for _name_i in _name_set
            ]
            self.ids.extend(
                _name_set_dict
            )

        if cfg.get('shuffle') != None:
            np.random.shuffle(self.ids)


        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if len(self.ids) == 0:
            raise Exception(
                "No files for style=[%s] found in %s" % (self.split, self.image_base_path)
            )
        
        print("Found {} {} images".format(len(self.ids), self.split))

    def __len__(self):
        """__len__"""
        if self.cfg.get('len', None) != None:
            # np.random.shuffle(self.ids)
            return self.cfg['len']
        return len(self.ids)

    def __getitem__(self, index):
        """__getitem__
        
        param: index
        """
        item_list = []
        id_dict = self.ids[index]

        id = id_dict['filename']
        img_path = id_dict['image_path']
        lbl_path = id_dict['gt_path']
            
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(e)
            print(img_path)
            id_dict = self.ids[(index+1)%len(self.ids)]
            id = id_dict['filename']
            img_path = id_dict['image_path']
            lbl_path = id_dict['gt_path']
            img = Image.open(img_path)

        lbl = Image.open(lbl_path)
        img = img.resize(self.img_size, Image.BILINEAR)
        lbl = lbl.resize(self.img_size, Image.NEAREST)
        img = np.asarray(img, dtype=np.uint8)

        lbl = np.asarray(lbl, dtype=np.uint8)

        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
            
        if self.augmentations!=None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
            
        item_list.extend([img, lbl, id])

        return item_list


    def encode_segmap(self, lbl):
        for _i in self.void_classes:
            lbl[lbl == _i] = self.ignore_index
        for _i in self.valid_classes:
            lbl[lbl == _i] = self.class_map[_i]
        return lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def transform(self, img, lbl):
        """transform

        img, lbl
        """
        img = np.array(img)
        # img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, self.img_size, "nearest", mode='F')
        lbl = lbl.astype(int)
        
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")    #TODO: compare the original and processed ones

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes): 
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

class MultiElemRandomSampler(data.sampler.Sampler):

    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self._num_samples = num_samples
        self._domains = [len(_ids) for _ids in data_source.ids]
        self.domain_num = len(data_source.root_Ds)

    @property
    def num_samples(self):
        domain_num = len(self._domains)
        if self._num_samples is None:
            return self._domains[0]
        return self._num_samples

    def __iter__(self):
        domain_size = self._domains
        iter_series = [
            torch.randperm(d_size).tolist()
            for d_size in self._domains
        ]
        #return iter(zip(iter_series))
        #pdb.set_trace()
        return zip(*iter_series)

    def __len__(self):
        return self.num_samples

if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(1024), RandomHorizontallyFlip(0.5)])

    local_path = "/home/qzha2506/remote/dataset/GTA5/"
    dst = GTA5_loader(local_path, split='train', is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=10)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        # import pdb;pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = raw_input()
        if a == "ex":
            break
        else:
            plt.close()
