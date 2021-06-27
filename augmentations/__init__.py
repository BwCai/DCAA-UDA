import logging
from augmentations.augmentations import *
from augmentations import augmentations_multilabels as ml

logger = logging.getLogger('ptsemseg')

key2aug = {'gamma': AdjustGamma,
           'hue': AdjustHue,
           'brightness': AdjustBrightness,
           'saturation': AdjustSaturation,
           'contrast': AdjustContrast,
           'rcrop': RandomCrop,
           'hflip': RandomHorizontallyFlip,
           'vflip': RandomVerticallyFlip,
           'scale': Scale,
           'rsize': RandomSized,
           'rsizecrop': RandomSizedCrop,
           'rotate': RandomRotate,
           'translate': RandomTranslate,
           'ccrop': CenterCrop,}

key2aug_multilabels = {'gamma': ml.AdjustGamma,
           'hue': ml.AdjustHue,
           'brightness': ml.AdjustBrightness,
           'saturation': ml.AdjustSaturation,
           'contrast': ml.AdjustContrast,
           'rcrop': ml.RandomCrop,
           'hflip': ml.RandomHorizontallyFlip,
           'vflip': ml.RandomVerticallyFlip,
           'scale': ml.Scale,
           'rsize': ml.RandomSized,
           'rsizecrop': ml.RandomSizedCrop,
           'rotate': ml.RandomRotate,
           'translate': ml.RandomTranslate,
           'ccrop': ml.CenterCrop,}

def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](aug_param))
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)

def get_composed_augmentations_multilabels(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug_multilabels[aug_key](aug_param))
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return ml.Compose(augmentations)


