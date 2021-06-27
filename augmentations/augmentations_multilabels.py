# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, masks):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            for _i in range(len(masks)):
                masks[_i] = Image.fromarray(masks[_i], mode="L")
            self.PIL2Numpy = True

        if img.size != masks[0].size:
            print (img.size, masks[0].size)
        assert img.size == masks[0].size
        for a in self.augmentations:
            img, masks = a(img, masks)
            # print(img.size)

        if self.PIL2Numpy:
            img = np.array(img)
            for _i in range(len(masks)):
                masks[_i] = np.array(masks[_i], dtype=np.uint8) 

        return img, masks


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, masks):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            for _i in range(len(masks)):
                masks[_i] = ImageOps.expand(masks[_i], border=self.padding, fill=0)

        assert img.size == masks[0].size
        w, h = img.size
        tw, th = self.size
        if w == tw and h == th:
            return img, masks
        if w < tw or h < th:
            for _i in range(len(masks)):
                masks[_i] = masks[_i].resize((tw, th), Image.NEAREST)
            return (
                img.resize((tw, th), Image.BILINEAR),
                masks,
            )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        
        for _i in range(len(masks)):
            masks[_i] = masks[_i].crop((x1, y1, x1 + tw, y1 + th))
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            masks,
        )

class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, masks):
        assert img.size == masks[0].size
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), masks


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, masks):
        assert img.size == masks[0].size
        return tf.adjust_saturation(img, 
                                    random.uniform(1 - self.saturation, 
                                                   1 + self.saturation)), masks


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, masks):
        assert img.size == masks[0].size
        return tf.adjust_hue(img, random.uniform(-self.hue, 
                                                  self.hue)), masks


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, masks):
        assert img.size == masks[0].size
        return tf.adjust_brightness(img, 
                                    random.uniform(1 - self.bf, 
                                                   1 + self.bf)), masks

class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, masks):
        assert img.size == masks[0].size
        return tf.adjust_contrast(img, 
                                  random.uniform(1 - self.cf, 
                                                 1 + self.cf)), masks

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, masks):
        assert img.size == masks[0].size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        for _i in range(len(masks)):
            masks[_i] = masks[_i].crop((x1, y1, x1 + tw, y1 + th))
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            masks,
        )


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, masks):
        if random.random() < self.p:
            for _i in range(len(masks)):
                masks[_i] = masks[_i].transpose(Image.FLIP_LEFT_RIGHT)
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                masks,
            )
        return img, masks


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, masks):
        if random.random() < self.p:
            for _i in range(len(masks)):
                masks[_i] = masks[_i].transpose(Image.FLIP_TOP_BOTTOM)
            return (
                img.transpose(Image.FLIP_TOP_BOTTOM),
                masks,
            )
        return img, masks


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, masks):
        assert img.size == mask.size
        for _i in range(len(masks)):
                masks[_i] = masks[_i].resize(self.size, Image.NEAREST)
        return (
            img.resize(self.size, Image.BILINEAR),
            masks,
        )


class RandomTranslate(object):
    def __init__(self, offset):
        self.offset = offset # tuple (delta_x, delta_y)

    def __call__(self, img, masks):
        assert img.size == mask.size
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])
        
        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0
        
        cropped_img = tf.crop(img, 
                              y_crop_offset, 
                              x_crop_offset, 
                              img.size[1]-abs(y_offset), 
                              img.size[0]-abs(x_offset))

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)
        
        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)
        
        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)
        
        for _i in range(len(masks)):
            masks[_i] = tf.affine(
                masks[_i],
                translate=(-x_offset, -y_offset),
                scale=1.0,
                angle=0.0,
                shear=0.0,
                fillcolor=250
            )
        return (
              tf.pad(cropped_img, 
                     padding_tuple, 
                     padding_mode='reflect'),
              masks)


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, masks):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        for _i in range(len(masks)):
            masks[_i] = tf.affine(masks[_i], 
                translate=(0, 0), 
                scale=1.0, 
                angle=rotate_degree, 
                resample=Image.NEAREST,
                fillcolor=250,
                shear=0.0
            )
        return (
            tf.affine(img, 
                      translate=(0, 0),
                      scale=1.0, 
                      angle=rotate_degree, 
                      resample=Image.BILINEAR,
                      fillcolor=(0, 0, 0),
                      shear=0.0),
            masks)



class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, masks):
        assert img.size == masks[0].size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, masks
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            
            for _i in range(len(masks)):
                masks[_i] = masks[_i].resize((ow, oh), Image.NEAREST)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                masks,
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)

            for _i in range(len(masks)):
                masks[_i] = masks[_i].resize((ow, oh), Image.NEAREST)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                masks,
            )

def MyScale(img, lbls, size):
    """scale

    img, lbl, longer size
    """
    if isinstance(img, np.ndarray):
        _img = Image.fromarray(img)
        for _i in range(len(lbls)):
            _lbls[_i] = Image.fromarray(lbls[_i])
    else:
        _img = img
        _lbls = lbls
    assert _img.size == _lbls[0].size
    # prop = 1.0 * _img.size[0]/_img.size[1]
    w, h = size
    # h = int(size / prop)
    _img = _img.resize((w, h), Image.BILINEAR)
    for _i in range(len(_lbls)):
        _lbls[_i] = np.array(_lbls[_i].resize((w, h), Image.NEAREST))
    return np.array(_img), _lbls

def Flip(img, lbls, prop):
    """
    flip img and lbl with probablity prop
    """
    if isinstance(img, np.ndarray):
        _img = Image.fromarray(img)
        for _i in range(len(lbls)):
            _lbls[_i] = Image.fromarray(lbls[_i])
    else:
        _img = img
        _lbls = lbls
    if random.random() < prop:
        _img.transpose(Image.FLIP_LEFT_RIGHT),
        for _i in range(len(_lbls)):
            _lbls[_i] = np.array(_lbls[_i].transpose(Image.FLIP_LEFT_RIGHT))
    return np.array(_img), _lbls

def MyRotate(img, lbls, degree):
    """
    img, lbl, degree
    randomly rotate clockwise or anti-clockwise
    """
    if isinstance(img, np.ndarray):
        _img = Image.fromarray(img)
        for _i in range(len(lbls)):
            _lbls[_i] = Image.fromarray(lbls[_i])
    else:
        _img = img
        _lbls = lbls
    _degree = random.random()*degree
    
    flags = -1
    if random.random() < 0.5:
        flags = 1
    _img = _img.rotate(_degree * flags)
    
    for _i in range(len(_lbls)):
        _lbls[_i] = np.array(_lbls[_i].rotate(_degree * flags))
    return np.array(_img), _lbls

class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, masks):
        assert img.size == masks[0].size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))

                for _i in range(len(masks)):
                    masks[_i] = masks[_i].crop((x1, y1, x1 + w, y1 + h)).resize((self.size, self.size), Image.NEAREST)
                
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    masks,
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, masks))


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, masks):
        assert img.size == masks[0].size

        prop = 1.0 * img.size[0] / img.size[1]
        w = int(random.uniform(0.5, 1.5) * self.size)
        h = int(w/prop)
        # h = int(random.uniform(0.5, 2) * self.size[1])

        img = img.resize((w, h), Image.BILINEAR)
        for _i in range(len(masks)):
            masks[_i] = masks[_i].resize((w, h), Image.NEAREST)

        return img, masks
        # return self.crop(*self.scale(img, mask))

