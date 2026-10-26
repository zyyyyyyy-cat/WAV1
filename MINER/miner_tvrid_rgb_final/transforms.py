"""Image augmentation transforms and transform pipeline builder.

Custom transforms used for MINER training on TVRID RGB track:
  - RandomCroping: resize 1.125x then crop back (spatial jitter)
  - RandomErasing: random rectangle erasure (Zhong et al., arXiv:1708.04896)
  - ColorJitter: random brightness/contrast/saturation/hue
  - RandomVerticalFlip: vertical flip (useful for top-view surveillance)
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import torch
from PIL import Image
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
import torchvision.transforms.functional as TF

from config import IMAGENET_MEAN, IMAGENET_STD, JITTER_BRIGHTNESS, JITTER_CONTRAST, JITTER_SATURATION, JITTER_HUE

# RandomCroping upscale factor (12.5% larger, then crop back)
_CROP_SCALE = 1.125

# RandomErasing fill values (CIFAR-10 per-channel means, standard practice)
_ERASING_MEAN = [0.4914, 0.4822, 0.4465]


class RandomCroping:
    """Resize 1.125x then random crop back to original size.

    With probability *p*, first upscale the image by 12.5% then randomly
    crop a region of the original size, simulating slight spatial jitter.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        if random.uniform(0, 1) >= self.p:
            return img
        new_w = int(round(width * _CROP_SCALE))
        new_h = int(round(height * _CROP_SCALE))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        x1 = int(round(random.uniform(0, new_w - width)))
        y1 = int(round(random.uniform(0, new_h - height)))
        return resized.crop((x1, y1, x1 + width, y1 + height))


class RandomErasing:
    """Randomly erase a rectangle in a CHW tensor image.

    Applied after ToTensor(). See Zhong et al., arXiv:1708.04896.
    """

    def __init__(
        self,
        probability: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.4,
        r1: float = 0.3,
        mean: List[float] = None,
    ):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean if mean is not None else list(_ERASING_MEAN)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) >= self.probability:
            return img
        for _ in range(100):
            area = img.size(1) * img.size(2)
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size(2) and h < img.size(1):
                x1 = random.randint(0, img.size(1) - h)
                y1 = random.randint(0, img.size(2) - w)
                for c in range(min(img.size(0), len(self.mean))):
                    img[c, x1 : x1 + h, y1 : y1 + w] = self.mean[c]
                return img
        return img


class ColorJitter:
    """Randomly change brightness, contrast, saturation and hue.

    Each factor is sampled independently; the four adjustments are applied
    in a random order. The whole transform fires with probability *p*.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        p: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        transforms = []
        if self.brightness > 0:
            bf = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            transforms.append(lambda img, _bf=bf: TF.adjust_brightness(img, _bf))
        if self.contrast > 0:
            cf = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            transforms.append(lambda img, _cf=cf: TF.adjust_contrast(img, _cf))
        if self.saturation > 0:
            sf = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            transforms.append(lambda img, _sf=sf: TF.adjust_saturation(img, _sf))
        if self.hue > 0:
            hf = random.uniform(-self.hue, self.hue)
            transforms.append(lambda img, _hf=hf: TF.adjust_hue(img, _hf))
        random.shuffle(transforms)
        for t in transforms:
            img = t(img)
        return img


class RandomVerticalFlip:
    """Vertically flip the given PIL Image with probability *p*.

    Particularly useful for top-view surveillance (TVRID) where vertical
    orientation is arbitrary.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


def build_transforms(cfg: dict) -> Tuple:
    """Build training and test transforms from a plain dict config.

    Args:
        cfg: Dict loaded from YAML with optional ``AUG`` and ``DATA`` sections.

    Returns:
        ``(transform_train, transform_test)`` -- torchvision Compose objects.
    """
    aug = cfg.get("AUG", {})
    size = cfg.get("DATA", {}).get("IMG_SIZE", 224)
    normalize = Normalize(IMAGENET_MEAN, IMAGENET_STD)

    train_transforms: list = [Resize((size, size))]

    if aug.get("RANDOM_CROP", 0) > 0:
        train_transforms.append(RandomCroping(p=aug["RANDOM_CROP"]))

    train_transforms.append(RandomHorizontalFlip(p=aug.get("HFLIP", 0.5)))

    if aug.get("VFLIP", 0) > 0:
        train_transforms.append(RandomVerticalFlip(p=aug["VFLIP"]))

    if aug.get("COLOR_JITTER", 0) > 0:
        train_transforms.append(
            ColorJitter(
                JITTER_BRIGHTNESS, JITTER_CONTRAST, JITTER_SATURATION, JITTER_HUE,
                p=aug["COLOR_JITTER"],
            )
        )

    train_transforms.extend([ToTensor(), normalize])

    if aug.get("RANDOM_ERASING", 0) > 0:
        train_transforms.append(RandomErasing(probability=aug["RANDOM_ERASING"]))

    transform_train = Compose(train_transforms)
    transform_test = Compose([Resize((size, size)), ToTensor(), normalize])
    return transform_train, transform_test
