"""Minimal measured visual input for Visual_Play.

The current product path deliberately begins with luminance only. More sensory
channels can be added later when the neural substrate has a reason to use them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualInput:
    brightness: np.ndarray


class VisualInputExtractor:
    """Convert a webcam frame into one normalized spatial brightness map."""

    def __init__(
        self,
        *,
        cols: int = 64,
        rows: int = 36,
        mirror: bool = True,
        scale_intermediate: Optional[Tuple[int, int]] = (640, 360),
    ) -> None:
        if cols < 1 or rows < 1:
            raise ValueError("cols and rows must be >= 1")
        self.cols = int(cols)
        self.rows = int(rows)
        self.mirror = bool(mirror)
        self.scale_intermediate = scale_intermediate

    def reset(self) -> None:
        """Reserved for future stateful sensory preprocessing."""

    def extract(self, frame: np.ndarray) -> VisualInput:
        if frame is None:
            raise ValueError("frame cannot be None")

        image = cv2.flip(frame, 1) if self.mirror else frame
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Expected grayscale or BGR frame, got {image.shape}")

        if self.scale_intermediate is not None:
            width, height = self.scale_intermediate
            if gray.shape[1] > width or gray.shape[0] > height:
                gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

        brightness = cv2.resize(
            gray,
            (self.cols, self.rows),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        brightness = np.clip(brightness / 255.0, 0.0, 1.0)
        return VisualInput(brightness=brightness)
