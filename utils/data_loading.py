import json
import logging
import numpy as np
import random
import torch
from PIL import Image, ImageEnhance, ImageOps
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.checkpoint_io import load_torch_state


class SegmentationTrainTransform:
    def __init__(self, mode: str = 'basic', seed: Optional[int] = None):
        self.mode = mode
        self.rng = random.Random(seed)

        if mode not in ('basic', 'strong'):
            raise ValueError("Unsupported augmentation mode '{}'. Use off, basic, or strong.".format(mode))

    def _chance(self, probability: float) -> bool:
        return self.rng.random() < probability

    def _uniform(self, low: float, high: float) -> float:
        return self.rng.uniform(low, high)

    def _affine(self, image: Image.Image, mask: Image.Image):
        if not self._chance(0.35):
            return image, mask

        max_shift = 0.04 if self.mode == 'basic' else 0.08
        max_scale_delta = 0.08 if self.mode == 'basic' else 0.15
        max_angle = 12 if self.mode == 'basic' else 25
        width, height = image.size

        angle = self._uniform(-max_angle, max_angle)
        scale = self._uniform(1.0 - max_scale_delta, 1.0 + max_scale_delta)
        tx = self._uniform(-max_shift, max_shift) * width
        ty = self._uniform(-max_shift, max_shift) * height

        image = image.transform(
            image.size,
            Image.AFFINE,
            (1.0 / scale, 0.0, -tx, 0.0, 1.0 / scale, -ty),
            resample=Image.BICUBIC,
            fillcolor=(255, 255, 255),
        ).rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        mask = mask.transform(
            mask.size,
            Image.AFFINE,
            (1.0 / scale, 0.0, -tx, 0.0, 1.0 / scale, -ty),
            resample=Image.NEAREST,
            fillcolor=0,
        ).rotate(angle, resample=Image.NEAREST, fillcolor=0)
        return image, mask

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        contrast_delta = 0.12 if self.mode == 'basic' else 0.22
        brightness_delta = 0.12 if self.mode == 'basic' else 0.22
        color_delta = 0.08 if self.mode == 'basic' else 0.16

        if self._chance(0.45):
            image = ImageEnhance.Contrast(image).enhance(self._uniform(1.0 - contrast_delta, 1.0 + contrast_delta))
        if self._chance(0.45):
            image = ImageEnhance.Brightness(image).enhance(self._uniform(1.0 - brightness_delta, 1.0 + brightness_delta))
        if self._chance(0.30):
            image = ImageEnhance.Color(image).enhance(self._uniform(1.0 - color_delta, 1.0 + color_delta))
        if self._chance(0.25):
            gamma = self._uniform(0.85, 1.15) if self.mode == 'basic' else self._uniform(0.75, 1.30)
            array = np.asarray(image).astype(np.float32) / 255.0
            array = np.power(np.clip(array, 0.0, 1.0), gamma)
            image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))
        return image

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.convert('RGB')
        mask = mask.convert('L')

        if self._chance(0.5):
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if self._chance(0.5):
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)
        if self._chance(0.5):
            k = self.rng.choice([1, 2, 3])
            image = image.rotate(90 * k, resample=Image.BICUBIC, expand=False)
            mask = mask.rotate(90 * k, resample=Image.NEAREST, expand=False)

        image, mask = self._affine(image, mask)
        image = self._color_jitter(image)
        return image, mask


def create_train_transform(mode: str = 'off', seed: Optional[int] = None):
    normalized = (mode or 'off').lower()
    if normalized in ('off', 'none', 'false', '0'):
        return None
    return SegmentationTrainTransform(mode=normalized, seed=seed)


def load_image(filename):
    ext = splitext(filename)[1].lower()
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(load_torch_state(filename).numpy())
    else:
        # Support for .jpg and .png via PIL
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    # Search for mask file with .png extension specifically
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.png'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        mask_dir: str,
        scale: float = 1.0,
        mask_suffix: str = '',
        transform=None,
    ):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transform

        # Filter for .jpg files to identify valid training samples
        self.ids = [splitext(file)[0] for file in listdir(images_dir) 
                    if isfile(join(images_dir, file)) and file.lower().endswith('.jpg') and not file.startswith('.')]
        
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your .jpg images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.mask_values = self._load_or_create_mask_values()
        logging.info(f'Unique mask values: {self.mask_values}')

    def _mask_value_cache_path(self) -> Path:
        return self.mask_dir / '.mask_values_cache.json'

    def _load_or_create_mask_values(self):
        cache_path = self._mask_value_cache_path()
        if cache_path.exists():
            try:
                with cache_path.open('r', encoding='utf-8') as cache_file:
                    payload = json.load(cache_file)

                if (
                    payload.get('mask_suffix') == self.mask_suffix
                    and payload.get('num_ids') == len(self.ids)
                    and isinstance(payload.get('mask_values'), list)
                ):
                    logging.info('Loaded cached mask values from %s', cache_path)
                    return payload['mask_values']
            except (OSError, ValueError, TypeError) as exc:
                logging.warning('Failed to read mask value cache %s: %s', cache_path, exc)

        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))

        try:
            with cache_path.open('w', encoding='utf-8') as cache_file:
                json.dump(
                    {
                        'mask_suffix': self.mask_suffix,
                        'num_ids': len(self.ids),
                        'mask_values': mask_values,
                    },
                    cache_file,
                    indent=2,
                )
        except OSError as exc:
            logging.warning('Failed to write mask value cache %s: %s', cache_path, exc)

        return mask_values

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            # Map unique mask values to class indices (0, 1, 2...)
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            # Transpose HWC to CHW format for PyTorch
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            # Normalize pixel values to [0, 1] range
            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        # Match specific extensions generated by prepare_data.py
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.png'))
        img_file = list(self.images_dir.glob(name + '.jpg'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, transform=None):
        # Set mask_suffix to empty since our script uses identical names for img and mask
        super().__init__(images_dir, mask_dir, scale, mask_suffix='', transform=transform)
