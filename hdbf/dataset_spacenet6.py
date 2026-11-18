from pathlib import Path as _HDBFPath
import sys as _hdbf_sys

_HDBF_ROOT = _HDBFPath(__file__).resolve().parents[1]
if str(_HDBF_ROOT) not in _hdbf_sys.path:
    _hdbf_sys.path.insert(0, str(_HDBF_ROOT))

import csv
import hashlib
import math
import os
import random

import numpy as np
import rasterio as rio
import torch
from torch.utils.data import Dataset


LABEL_FIELDS = {"mask", "contour", "height", "valid_height", "instance"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def split_metadata(directory):
    result, groups = {}, {}
    for split in ("train", "val", "test"):
        path = os.path.join(directory, split + ".csv")
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        keys = [r["key"] for r in rows]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Empty or duplicate split: " + path)
        groups[split] = set(keys)
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        result[split] = dict(count=len(keys), sha256=digest)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[a] & groups[b]:
            raise ValueError("Cross-split tile IDs: " + a + "/" + b)
    return result


def _normalize_path(path):
    return os.path.normpath(path.replace("\\", "/"))


class JointRandomFlipRotate:
    def __call__(self, arrays):
        if random.random() < 0.5:
            axis = random.choice([-1, -2])
            arrays = [np.flip(array, axis=axis).copy() for array in arrays]
        k = random.choice([0, 1, 2, 3])
        return [np.rot90(array, k=k, axes=(-2, -1)).copy() for array in arrays]


class SpaceNet6HeightDataset(Dataset):
    """SpaceNet6 RGB-SAR data with dense pixel-level DSM-minus-DTM heights.

    Training samples are jointly cropped to ``crop_size``. Validation and test
    samples remain at their native 900 x 900 resolution so each source pixel is
    evaluated exactly once.
    """

    def __init__(
        self,
        processed_dir,
        split="train",
        train=True,
        crop_size=512,
        crops_per_image=1,
        sar_clip=60.0,
        sar_normalization="linear",
        min_building_pixels=64,
        positive_crop_probability=0.7,
        crop_attempts=10,
        targeted_crop_probability=0.0,
        targeted_height_min=8.0,
        targeted_height_max=20.0,
        targeted_area_min_m2=500.0,
        source_dir=None,
        normalize_rgb=False,
        split_dir=None,
    ):
        super().__init__()
        self.processed_dir = os.path.abspath(processed_dir)
        self.source_dir = os.path.abspath(
            source_dir or os.path.dirname(self.processed_dir)
        )
        self.train = bool(train)
        self.normalize_rgb = bool(normalize_rgb)
        self.crop_size = int(crop_size) if self.train and crop_size else None
        self.crops_per_image = max(int(crops_per_image), 1) if self.train else 1
        self.sar_clip = float(sar_clip)
        self.sar_normalization = str(sar_normalization).lower()
        self.min_building_pixels = max(int(min_building_pixels), 0)
        self.positive_crop_probability = float(positive_crop_probability)
        self.crop_attempts = max(int(crop_attempts), 1)
        self.targeted_crop_probability = float(targeted_crop_probability)
        self.targeted_height_min = float(targeted_height_min)
        self.targeted_height_max = float(targeted_height_max)
        self.targeted_area_min_m2 = float(targeted_area_min_m2)
        self.augmentation = JointRandomFlipRotate() if self.train else None

        if self.sar_clip <= 0:
            raise ValueError("sar_clip must be positive")
        if self.sar_normalization not in {"linear", "log"}:
            raise ValueError("sar_normalization must be 'linear' or 'log'")
        if not 0.0 <= self.positive_crop_probability <= 1.0:
            raise ValueError("positive_crop_probability must be in [0, 1]")
        if not 0.0 <= self.targeted_crop_probability <= 1.0:
            raise ValueError("targeted_crop_probability must be in [0, 1]")
        if self.targeted_height_min <= 0:
            raise ValueError("targeted_height_min must be positive")
        if self.targeted_height_max <= self.targeted_height_min:
            raise ValueError("targeted_height_max must exceed targeted_height_min")
        if self.targeted_area_min_m2 <= 0:
            raise ValueError("targeted_area_min_m2 must be positive")
        if self.crop_size is not None and self.crop_size % 32 != 0:
            raise ValueError("crop_size must be divisible by 32")

        manifest = os.path.join(
            split_dir or os.path.join(self.processed_dir, "splits"), f"{split}.csv"
        )
        if not os.path.isfile(manifest):
            raise FileNotFoundError(f"SpaceNet6 split manifest not found: {manifest}")
        with open(manifest, "r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise RuntimeError(f"SpaceNet6 split is empty: {manifest}")

        required = {"key", "rgb", "sar", *LABEL_FIELDS}
        missing = required.difference(self.rows[0])
        if missing:
            raise KeyError(f"Missing columns in {manifest}: {sorted(missing)}")

        # Resolve one complete record immediately to fail before training starts.
        for field in ("rgb", "sar", *sorted(LABEL_FIELDS)):
            path = self._resolve_path(self.rows[0][field], field)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Cannot resolve {field} path: {path}")
        instance_path = self._resolve_path(self.rows[0]["instance"], "instance")
        with rio.open(instance_path) as source:
            transform = source.transform
        self.pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)

    def __len__(self):
        return len(self.rows) * self.crops_per_image

    def _resolve_path(self, stored_path, field):
        normalized = _normalize_path(stored_path)
        if os.path.isfile(normalized):
            return normalized

        basename = os.path.basename(normalized)
        if field in LABEL_FIELDS:
            return os.path.join(self.processed_dir, "labels", field, basename)

        parts = normalized.replace("\\", "/").split("/")
        try:
            marker = parts.index("AOI_11_Rotterdam")
        except ValueError as error:
            raise FileNotFoundError(
                f"Cannot make manifest path portable: {stored_path}"
            ) from error
        return os.path.join(self.source_dir, *parts[marker:])

    @staticmethod
    def _read(path, dtype):
        with rio.open(path) as source:
            array = source.read().astype(dtype, copy=False)
        return array

    def _load_record(self, row):
        rgb = self._read(self._resolve_path(row["rgb"], "rgb"), np.float32)[:3]
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
        if rgb.max() > 2.0:
            rgb /= 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
        if self.normalize_rgb:
            rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        sar = self._read(self._resolve_path(row["sar"], "sar"), np.float32)
        if sar.shape[0] != 4:
            raise ValueError(f"Expected four SpaceNet6 SAR bands, got {sar.shape[0]}")
        sar = np.nan_to_num(sar, nan=0.0, posinf=self.sar_clip, neginf=0.0)
        sar = np.clip(sar, 0.0, self.sar_clip)
        if self.sar_normalization == "log":
            # Legacy mode for checkpoints trained before linear dB scaling.
            sar = np.log1p(sar) / math.log1p(self.sar_clip)
        else:
            # SpaceNet6 SAR-Intensity is already expressed in decibels.
            sar = sar / self.sar_clip

        mask = self._read(self._resolve_path(row["mask"], "mask"), np.uint8)
        contour = self._read(self._resolve_path(row["contour"], "contour"), np.uint8)
        height = self._read(self._resolve_path(row["height"], "height"), np.float32)
        valid_height = self._read(
            self._resolve_path(row["valid_height"], "valid_height"), np.uint8
        )
        instance = self._read(self._resolve_path(row["instance"], "instance"), np.int64)

        mask = (mask > 0).astype(np.float32)
        contour = (contour > 0).astype(np.float32)
        valid_height = (valid_height > 0).astype(np.float32)
        height = np.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)
        height = np.clip(height, 0.0, None).astype(np.float32, copy=False)
        instance = np.where(mask > 0, instance, 0).astype(np.int64, copy=False)

        shapes = {
            array.shape[-2:]
            for array in (rgb, sar, mask, contour, height, valid_height, instance)
        }
        if len(shapes) != 1:
            raise ValueError(
                f"Mismatched raster sizes for {row['key']}: {sorted(shapes)}"
            )
        return rgb, sar, mask, contour, height, valid_height, instance

    def _targeted_crop_origin(self, arrays, crop):
        height_map = arrays[4][0]
        valid_height = arrays[5][0] > 0
        instance_map = arrays[6][0]
        positive = instance_map > 0
        if not positive.any():
            return None

        candidates = []
        instance_ids, counts = np.unique(instance_map[positive], return_counts=True)
        for instance_id, count in zip(instance_ids, counts):
            area_m2 = float(count) * self.pixel_area_m2
            if area_m2 < self.targeted_area_min_m2:
                continue
            reliable = (
                (instance_map == instance_id)
                & valid_height
                & np.isfinite(height_map)
                & (height_map > 0.1)
            )
            if not reliable.any():
                continue
            mean_height = float(height_map[reliable].mean())
            if not (self.targeted_height_min <= mean_height < self.targeted_height_max):
                continue
            candidates.append(int(instance_id))

        if not candidates:
            return None
        selected_id = random.choice(candidates)
        rows, columns = np.where(instance_map == selected_id)
        center_row = int(round((int(rows.min()) + int(rows.max())) / 2.0))
        center_column = int(round((int(columns.min()) + int(columns.max())) / 2.0))
        jitter = max(crop // 10, 1)
        center_row += random.randint(-jitter, jitter)
        center_column += random.randint(-jitter, jitter)

        height, width = instance_map.shape
        top = min(max(center_row - crop // 2, 0), height - crop)
        left = min(max(center_column - crop // 2, 0), width - crop)
        return top, left

    def _random_crop(self, arrays):
        crop = self.crop_size
        height, width = arrays[0].shape[-2:]
        if crop is None:
            return arrays, (0, 0, height, width)
        if height < crop or width < crop:
            raise ValueError(f"crop_size={crop} exceeds source size {(height, width)}")

        if random.random() < self.targeted_crop_probability:
            origin = self._targeted_crop_origin(arrays, crop)
            if origin is not None:
                top, left = origin
                cropped = [
                    array[..., top : top + crop, left : left + crop] for array in arrays
                ]
                return cropped, (top, left, crop, crop)

        mask = arrays[2]
        require_building = (
            self.min_building_pixels > 0
            and mask.sum() > 0
            and random.random() < self.positive_crop_probability
        )
        top = left = 0
        attempts = self.crop_attempts if require_building else 1
        for _ in range(attempts):
            top = random.randint(0, height - crop)
            left = random.randint(0, width - crop)
            candidate = mask[..., top : top + crop, left : left + crop]
            if not require_building or candidate.sum() >= self.min_building_pixels:
                break
        cropped = [array[..., top : top + crop, left : left + crop] for array in arrays]
        return cropped, (top, left, crop, crop)

    def __getitem__(self, index):
        row = self.rows[index // self.crops_per_image]
        arrays = list(self._load_record(row))
        arrays, crop_window = self._random_crop(arrays)
        if self.augmentation is not None:
            arrays = self.augmentation(arrays)
        rgb, sar, mask, contour, height, valid_height, instance = arrays

        return {
            "filename": row["key"],
            "crop_window": torch.tensor(crop_window, dtype=torch.int32),
            "rgb": torch.from_numpy(rgb.copy()).float(),
            "sar": torch.from_numpy(sar.copy()).float(),
            "mask": torch.from_numpy(mask.copy()).float(),
            "contour": torch.from_numpy(contour.copy()).float(),
            "height": torch.from_numpy(height.copy()).float(),
            "valid_height": torch.from_numpy(valid_height.copy()).float(),
            "instance": torch.from_numpy(instance.copy()).long(),
        }
