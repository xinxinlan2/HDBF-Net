from pathlib import Path as _HDBFPath
import sys as _hdbf_sys

_HDBF_ROOT = _HDBFPath(__file__).resolve().parents[1]
if str(_HDBF_ROOT) not in _hdbf_sys.path:
    _hdbf_sys.path.insert(0, str(_HDBF_ROOT))

import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw
import rasterio as rio
import torch
from torch.utils.data import Dataset


def polygon_to_mask(polygons, height, width):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) < 6:
            continue
        xy = []
        for i in range(0, len(poly), 2):
            x = min(max(float(poly[i]), 0.0), width - 1.0)
            y = min(max(float(poly[i + 1]), 0.0), height - 1.0)
            xy.append((x, y))
        if len(xy) >= 3:
            draw.polygon(xy, outline=1, fill=1)
    return np.asarray(mask, dtype=np.uint8)


def _decode_compressed_rle_counts(encoded):
    if isinstance(encoded, str):
        try:
            encoded = encoded.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError(
                "COCO RLE counts must contain ASCII characters."
            ) from error
    counts = []
    position = 0
    # COCO uses signed 5-bit groups, then a two-run delta from run index 3.
    # Format reference: cocodataset/cocoapi, common/maskApi.c, rleFrString.
    while position < len(encoded):
        value, shift = 0, 0
        while True:
            if position >= len(encoded):
                raise ValueError("Truncated compressed COCO RLE counts.")
            code = encoded[position] - 48
            position += 1
            if not 0 <= code <= 63:
                raise ValueError("Invalid compressed COCO RLE character.")
            value |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    value -= 1 << shift
                break
            if shift >= 35:
                raise ValueError("Compressed COCO RLE run is too long.")
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts


def decode_rle(segmentation, height, width):
    """Decode compressed or uncompressed COCO RLE without extra packages."""
    if tuple(segmentation.get("size", ())) != (height, width):
        raise ValueError("COCO RLE size does not match the label grid.")
    if height <= 0 or width <= 0:
        raise ValueError("COCO RLE label dimensions must be positive.")
    counts = segmentation.get("counts")
    if isinstance(counts, (str, bytes, bytearray)):
        counts = _decode_compressed_rle_counts(counts)
    elif not isinstance(counts, list):
        raise ValueError(
            "COCO RLE counts must be an integer list or compressed string."
        )
    total = height * width
    flat = np.zeros(total, dtype=np.uint8)
    position = 0
    for index, count in enumerate(counts):
        if isinstance(count, (bool, np.bool_)) or not isinstance(
            count, (int, np.integer)
        ):
            raise ValueError("COCO RLE runs must be integers.")
        count = int(count)
        if count < 0 or count > total - position:
            raise ValueError("COCO RLE run exceeds the label grid.")
        end = position + count
        if index % 2:
            flat[position:end] = 1
        position = end
    if position != total:
        raise ValueError("COCO RLE runs do not cover the complete label grid.")
    return flat.reshape((height, width), order="F")


def instances_to_targets(instances, height, width):
    """Rasterize COCO polygons and RLE with one integer id per annotation."""
    mask_image = Image.new("L", (width, height), 0)
    instance_image = Image.new("I", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask_image)
    instance_draw = ImageDraw.Draw(instance_image)
    next_instance_id = 1
    for segmentation in instances:
        if isinstance(segmentation, dict):
            building = decode_rle(segmentation, height, width)
            if building.any():
                selection = Image.fromarray(building * 255)
                mask_image.paste(1, (0, 0), selection)
                instance_image.paste(next_instance_id, (0, 0), selection)
                next_instance_id += 1
            continue
        drew_instance = False
        for poly in segmentation:
            if len(poly) < 6:
                continue
            xy = []
            for i in range(0, len(poly), 2):
                x = min(max(float(poly[i]), 0.0), width - 1.0)
                y = min(max(float(poly[i + 1]), 0.0), height - 1.0)
                xy.append((x, y))
            if len(xy) < 3:
                continue
            mask_draw.polygon(xy, outline=1, fill=1)
            instance_draw.polygon(xy, outline=next_instance_id, fill=next_instance_id)
            drew_instance = True
        if drew_instance:
            next_instance_id += 1
    return np.asarray(mask_image, dtype=np.uint8), np.asarray(
        instance_image, dtype=np.int32
    )


def build_annotation_map(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id_to_name = {img["id"]: img["file_name"] for img in coco.get("images", [])}
    annotations = {}
    for ann in coco.get("annotations", []):
        filename = id_to_name.get(ann.get("image_id"))
        segmentation = ann.get("segmentation", [])
        if filename is None:
            continue
        annotations.setdefault(filename, []).append(segmentation)
    return annotations


def mask_to_contour(mask, width=2):
    mask = mask.astype(np.uint8)
    padded = np.pad(mask, width, mode="constant", constant_values=0)
    dilated = np.zeros_like(mask)
    eroded = np.ones_like(mask)
    kernel_size = 2 * width + 1
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            crop = padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
            dilated = np.maximum(dilated, crop)
            eroded = np.minimum(eroded, crop)
    return (dilated - eroded).astype(np.uint8)


class JointRandomFlipRotate:
    def __call__(self, arrays):
        if random.random() < 0.5:
            axis = random.choice([-1, -2])
            return [np.flip(array, axis=axis).copy() for array in arrays]
        k = random.choice([0, 1, 2, 3])
        return [np.rot90(array, k=k, axes=(-2, -1)).copy() for array in arrays]


class RGBPhotometricJitter:
    def __init__(self, probability=0.0, strength=0.1):
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("RGB jitter probability must be in [0, 1].")
        if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("RGB jitter strength must be in [0, 1].")
        self.probability = float(probability)
        self.strength = float(strength)

    def __call__(self, rgb):
        if self.probability == 0.0 or self.strength == 0.0:
            return rgb
        if random.random() >= self.probability:
            return rgb
        brightness = np.float32(
            random.uniform(1.0 - self.strength, 1.0 + self.strength)
        )
        contrast = np.float32(random.uniform(1.0 - self.strength, 1.0 + self.strength))
        image = rgb * brightness
        center = np.float32(image.mean(dtype=np.float64))
        return np.clip((image - center) * contrast + center, 0.0, 1.0).astype(
            np.float32
        )


class DFC2023Dataset(Dataset):
    def __init__(
        self,
        base_dir,
        split_file,
        train=True,
        sar_clip=0.5,
        contour_width=2,
        rgb_jitter_probability=0.0,
        rgb_jitter_strength=0.1,
        rgb_normalization="none",
    ):
        super().__init__()
        if rgb_normalization not in ("none", "imagenet"):
            raise ValueError("RGB normalization must be 'none' or 'imagenet'.")
        self.rgb_normalization = rgb_normalization
        self.train = train
        self.sar_clip = sar_clip
        train_root = os.path.join(base_dir, "track2", "train")
        self.rgb_dir = os.path.join(train_root, "rgb")
        self.sar_dir = os.path.join(train_root, "sar")
        self.dsm_dir = os.path.join(train_root, "dsm")
        json_path = os.path.join(base_dir, "track2", "buildings_only_train.json")
        self.annotation_map = build_annotation_map(json_path)
        self.contour_width = contour_width
        self.augmentation = JointRandomFlipRotate() if train else None
        jitter = RGBPhotometricJitter(rgb_jitter_probability, rgb_jitter_strength)
        self.rgb_jitter = jitter if train and jitter.probability > 0.0 else None

        with open(split_file, "r", encoding="utf-8") as f:
            self.files = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.files)

    def _read_rgb(self, filename):
        with rio.open(os.path.join(self.rgb_dir, filename)) as src:
            rgb = src.read()[:3].astype(np.float32)
        if rgb.max() > 2.0:
            rgb /= 255.0
        return np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

    def _read_sar(self, filename):
        with rio.open(os.path.join(self.sar_dir, filename)) as src:
            sar = src.read()[:1].astype(np.float32)
        sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0)
        if self.sar_clip is not None:
            sar = np.clip(sar, 0.0, self.sar_clip) / self.sar_clip
        return sar

    def _read_height(self, filename):
        with rio.open(os.path.join(self.dsm_dir, filename)) as src:
            height = src.read(1).astype(np.float32)
            valid = (src.read_masks(1) > 0) & np.isfinite(height)
            if src.nodata is not None:
                valid &= height != src.nodata
        height = np.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)
        height[~valid] = 0.0
        return np.clip(height, 0.0, None)[None], valid[None]

    def __getitem__(self, index):
        filename = self.files[index]
        rgb = self._read_rgb(filename)
        sar = self._read_sar(filename)
        height, valid_height = self._read_height(filename)
        mask, instance = instances_to_targets(
            self.annotation_map.get(filename, []),
            height.shape[-2],
            height.shape[-1],
        )
        mask = mask[None].astype(np.float32)
        instance = instance[None]
        contour = mask_to_contour(mask[0], self.contour_width)[None].astype(np.float32)

        if self.augmentation is not None:
            rgb, sar, mask, contour, height, valid_height, instance = self.augmentation(
                [rgb, sar, mask, contour, height, valid_height, instance]
            )
        if self.rgb_jitter is not None:
            rgb = self.rgb_jitter(rgb)
        # Photometric augmentation operates on RGB intensities before normalization.
        if self.rgb_normalization == "imagenet":
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            rgb = (rgb - mean) / std

        return {
            "filename": filename,
            "rgb": torch.from_numpy(rgb.copy()).float(),
            "sar": torch.from_numpy(sar.copy()).float(),
            "mask": torch.from_numpy(mask.copy()).float(),
            "contour": torch.from_numpy(contour.copy()).float(),
            "height": torch.from_numpy(height.copy()).float(),
            "valid_height": torch.from_numpy(valid_height.copy()).bool(),
            "instance": torch.from_numpy(instance.copy()).long(),
        }
