"""
vision_features.py

Fixed visual feature extraction for Visual_Play.

Converts webcam frames into five compact spatial maps:
brightness, contrast, motion, horizontal edges, and vertical edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VisionFeatures:
    """Container for one frame's extracted visual features."""

    CHANNEL_NAMES: ClassVar[Tuple[str, ...]] = (
        "brightness",
        "contrast",
        "motion",
        "horizontal",
        "vertical",
    )

    brightness: np.ndarray
    contrast: np.ndarray
    motion: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray

    @property
    def tensor(self) -> np.ndarray:
        return np.stack(
            [
                self.brightness,
                self.contrast,
                self.motion,
                self.horizontal,
                self.vertical,
            ],
            axis=0,
        )

    @property
    def vector(self) -> np.ndarray:
        return self.tensor.reshape(-1)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (
            len(self.CHANNEL_NAMES),
            self.brightness.shape[0],
            self.brightness.shape[1],
        )

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "motion": self.motion,
            "horizontal": self.horizontal,
            "vertical": self.vertical,
        }


class VisionFeatureExtractor:
    """Extract simple fixed visual signals from webcam frames."""

    def __init__(
        self,
        cols: int = 64,
        rows: int = 32,
        contrast_kernel: int = 9,
        contrast_gain: float = 6.0,
        motion_gain: float = 6.5,
        edge_gain: float = 3.5,
        motion_threshold: float = 0.02,
        motion_decay: float = 0.70,
        scale_intermediate: Optional[Tuple[int, int]] = (640, 360),
        mirror: bool = True,
        normalize: bool = True,
    ) -> None:
        if cols < 1 or rows < 1:
            raise ValueError("cols and rows must be >= 1")

        if contrast_kernel < 3:
            contrast_kernel = 3
        if contrast_kernel % 2 == 0:
            contrast_kernel += 1

        self.cols = cols
        self.rows = rows
        self.contrast_kernel = contrast_kernel
        self.contrast_gain = float(contrast_gain)
        self.motion_gain = float(motion_gain)
        self.edge_gain = float(edge_gain)
        self.motion_threshold = float(motion_threshold)
        self.motion_decay = float(np.clip(motion_decay, 0.0, 0.99))
        self.scale_intermediate = scale_intermediate
        self.mirror = mirror
        self.normalize = normalize

        self._previous_gray: Optional[np.ndarray] = None
        self._motion_accumulator: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._previous_gray = None
        self._motion_accumulator = None

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            raise ValueError("frame cannot be None")

        if self.mirror:
            frame = cv2.flip(frame, 1)

        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(
                f"Expected grayscale or BGR frame, got shape {frame.shape}"
            )

        if self.scale_intermediate is not None:
            inter_w, inter_h = self.scale_intermediate
            if gray.shape[1] > inter_w or gray.shape[0] > inter_h:
                gray = cv2.resize(
                    gray,
                    (inter_w, inter_h),
                    interpolation=cv2.INTER_AREA,
                )

        return gray.astype(np.float32)

    def _pool(self, feature_map: np.ndarray, gain: float = 1.0) -> np.ndarray:
        pooled = cv2.resize(
            feature_map,
            (self.cols, self.rows),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

        if self.normalize:
            pooled = np.clip((pooled / 255.0) * gain, 0.0, 1.0)
        return pooled

    def extract(self, frame: np.ndarray) -> VisionFeatures:
        gray = self._to_gray(frame)

        brightness_map = gray

        k = self.contrast_kernel
        mean = cv2.blur(gray, (k, k))
        mean_sq = cv2.blur(gray * gray, (k, k))
        variance = np.maximum(mean_sq - (mean * mean), 0.0)
        contrast_map = np.sqrt(variance) * self.contrast_gain

        if self._previous_gray is None or self._previous_gray.shape != gray.shape:
            raw_motion = np.zeros_like(gray)
            self._motion_accumulator = np.zeros_like(gray)
        else:
            diff = np.abs(gray - self._previous_gray)
            thresh_val = self.motion_threshold * 255.0
            raw_motion = np.where(
                diff >= thresh_val,
                diff * self.motion_gain,
                0.0,
            )

            if self.motion_decay > 0.0 and self._motion_accumulator is not None:
                self._motion_accumulator = np.maximum(
                    raw_motion,
                    self._motion_accumulator * self.motion_decay,
                )
                raw_motion = self._motion_accumulator
            else:
                self._motion_accumulator = raw_motion.copy()

        self._previous_gray = gray.copy()

        gx = cv2.Sobel(
            gray,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
            scale=0.25,
        )
        gy = cv2.Sobel(
            gray,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
            scale=0.25,
        )

        # A vertical intensity gradient (gy) highlights horizontal structure;
        # a horizontal intensity gradient (gx) highlights vertical structure.
        horizontal_map = np.abs(gy) * self.edge_gain
        vertical_map = np.abs(gx) * self.edge_gain

        return VisionFeatures(
            brightness=self._pool(brightness_map),
            contrast=self._pool(contrast_map),
            motion=self._pool(raw_motion),
            horizontal=self._pool(horizontal_map),
            vertical=self._pool(vertical_map),
        )
