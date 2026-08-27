"""
vision_features.py

Fixed visual feature extraction for Visual_Play.

Purpose:
    Convert webcam frames into a compact, standardized set of basic visual signals.
    This module serves as a pure sensory preprocessing layer (no learned weights).

Output feature maps:
    brightness   - pooled luminance
    contrast     - local intensity variation (adaptive standard deviation)
    motion       - frame-to-frame change (with dynamic sensitivity & persistence)
    horizontal   - horizontal edge / gradient energy (calibrated dynamic range)
    vertical     - vertical edge / gradient energy (calibrated dynamic range)

Default output:
    30 x 18 spatial grid x 5 feature channels = 2,700 signals (configurable)
"""

from __future__ import annotations

import time
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
        """Returns shape: (5, rows, cols)."""
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
        """Flattened feature vector."""
        return self.tensor.reshape(-1)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape tuple: (5, rows, cols)."""
        return (len(self.CHANNEL_NAMES), self.brightness.shape[0], self.brightness.shape[1])

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "motion": self.motion,
            "horizontal": self.horizontal,
            "vertical": self.vertical,
        }


class VisionFeatureExtractor:
    """
    Extracts simple, fixed visual signals from webcam frames.

    Features calibrated channel gain, adaptive percentile scaling,
    temporal motion decay, and sub-millisecond execution.
    """

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
        """Forget previous frames used for motion detection."""
        self._previous_gray = None
        self._motion_accumulator = None

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale, apply mirror, and pre-scale."""
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

        # Pre-scale to intermediate size for 10-15x faster filter processing
        if self.scale_intermediate is not None:
            inter_w, inter_h = self.scale_intermediate
            if gray.shape[1] > inter_w or gray.shape[0] > inter_h:
                gray = cv2.resize(gray, (inter_w, inter_h), interpolation=cv2.INTER_AREA)

        return gray.astype(np.float32)

    def _pool(self, feature_map: np.ndarray, gain: float = 1.0) -> np.ndarray:
        """Area-pool a feature map into the compact spatial grid with calibrated gain."""
        pooled = cv2.resize(
            feature_map,
            (self.cols, self.rows),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

        if self.normalize:
            pooled = np.clip((pooled / 255.0) * gain, 0.0, 1.0)

        return pooled

    def extract(self, frame: np.ndarray) -> VisionFeatures:
        """
        Extract features from one frame.

        Parameters
        ----------
        frame:
            BGR webcam frame or grayscale frame.

        Returns
        -------
        VisionFeatures
            Five pooled feature maps with vivid dynamic range.
        """
        gray = self._to_gray(frame)

        # 1. Brightness / luminance
        brightness_map = gray

        # 2. Local contrast: standard deviation inside local neighborhood with boost
        k = self.contrast_kernel
        mean = cv2.blur(gray, (k, k))
        mean_sq = cv2.blur(gray * gray, (k, k))
        variance = np.maximum(mean_sq - (mean * mean), 0.0)
        contrast_map = np.sqrt(variance) * self.contrast_gain

        # 3. Motion: frame-to-frame change with high sensitivity & temporal glow
        if self._previous_gray is None or self._previous_gray.shape != gray.shape:
            raw_motion = np.zeros_like(gray)
            self._motion_accumulator = np.zeros_like(gray)
        else:
            diff = np.abs(gray - self._previous_gray)
            thresh_val = self.motion_threshold * 255.0
            raw_motion = np.where(diff >= thresh_val, diff * self.motion_gain, 0.0)

            if self.motion_decay > 0.0 and self._motion_accumulator is not None:
                self._motion_accumulator = np.maximum(
                    raw_motion,
                    self._motion_accumulator * self.motion_decay,
                )
                raw_motion = self._motion_accumulator
            else:
                self._motion_accumulator = raw_motion.copy()

        self._previous_gray = gray.copy()
        motion_map = raw_motion

        # 4 & 5. Directional gradient energy with calibrated scale
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=0.25)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=0.25)

        horizontal_map = np.abs(gx) * self.edge_gain
        vertical_map = np.abs(gy) * self.edge_gain

        return VisionFeatures(
            brightness=self._pool(brightness_map, gain=1.0),
            contrast=self._pool(contrast_map, gain=1.0),
            motion=self._pool(motion_map, gain=1.0),
            horizontal=self._pool(horizontal_map, gain=1.0),
            vertical=self._pool(vertical_map, gain=1.0),
        )


if __name__ == "__main__":
    # Interactive visual sensory suite
    cols, rows = 64, 32
    extractor = VisionFeatureExtractor(
        cols=cols,
        rows=rows,
        mirror=True,
        motion_decay=0.75,
        contrast_gain=7.0,
        motion_gain=8.0,
        edge_gain=4.5,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise SystemExit("Could not access webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Visual Play - Sensory Engine", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Visual Play - Sensory Engine", 1400, 720)

    # Distinct Neon Signatures (BGR) for each channel
    NEON_COLORS = {
        "brightness": np.array([255, 215, 60], dtype=np.float32),   # Solar Gold / Cyan
        "contrast":   np.array([255, 60, 230], dtype=np.float32),   # Electric Magenta
        "motion":     np.array([40, 90, 255], dtype=np.float32),    # Blazing Solar Flame
        "horizontal": np.array([50, 255, 130], dtype=np.float32),   # Matrix Emerald
        "vertical":   np.array([255, 190, 30], dtype=np.float32),   # Laser Electric Blue
    }

    THEMES = ["Neon Cyber-Vision", "Inferno Fire", "Plasma", "Turbo Spectrum", "Monochrome Noir"]
    def open_camera():
        c = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not c.isOpened():
            c = cv2.VideoCapture(0)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return c

    cap = open_camera()
    if cap is None or not cap.isOpened():
        raise SystemExit("Could not access webcam.")

    cv2.namedWindow("Visual Play - Sensory Engine", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Visual Play - Sensory Engine", 1380, 800)

    theme_idx = 0
    view_mode = "split"  # "split" (5 tiles) or "fusion" (composite HUD)
    sensitivity = 1.0
    camera_on = True

    def toggle_camera():
        global cap, camera_on
        if camera_on:
            if cap is not None:
                cap.release()
                cap = None
            camera_on = False
            print(">> CAMERA PHYSICALLY DISCONNECTED (Hardware green light OFF)")
        else:
            cap = open_camera()
            if cap is not None and cap.isOpened():
                camera_on = True
                print(">> CAMERA PHYSICALLY CONNECTED (Hardware green light ON)")
            else:
                camera_on = False
                print(">> Error: Could not re-open camera device.")

    channel_meta = [
        ("1. Brightness", "brightness", (255, 215, 60)),
        ("2. Contrast", "contrast", (255, 60, 230)),
        ("3. Motion", "motion", (40, 90, 255)),
        ("4. Horizontal", "horizontal", (50, 255, 130)),
        ("5. Vertical", "vertical", (255, 190, 30)),
    ]

    print("=" * 65)
    print("       VISUAL PLAY - SENSORY FEATURE EXTRACTOR")
    print("=" * 65)
    print("Controls:")
    print("  [k], [SPACE], [p] : Physically Connect / Disconnect Camera (Killswitch)")
    print("  [v]         : Toggle View Mode (5-Channel Panorama <-> Cyber Fusion HUD)")
    print("  [c]         : Cycle Visual Themes (Neon, Inferno, Plasma, Turbo, Noir)")
    print("  [+] / [-]   : Step Spatial Grid Density")
    print("  [ [ ] / [ ] ] : Adjust Sensory Sensitivity Gain")
    print("  [m]         : Toggle Motion Persistence Trace")
    print("  [x]         : Toggle Mirror Orientation")
    print("  [q]         : Quit")
    print("=" * 65)

    prev_t = time.time()
    fps = 0.0

        f_vert = np.clip(features.vertical * sensitivity, 0.0, 1.0)

        f_dict = {
            "brightness": f_bright,
            "contrast": f_cont,
            "motion": f_mot,
            "horizontal": f_horiz,
            "vertical": f_vert,
        }

        theme = THEMES[theme_idx]

        if view_mode == "split":
            # 5-Channel Panoramic View
            tile_w, tile_h = 276, 180
            tiles = []

            for label, key_name, banner_col in channel_meta:
                fmap = f_dict[key_name]
                uint8_f = (fmap * 255.0).astype(np.uint8)

                if theme == "Neon Cyber-Vision":
                    # Custom Glowing Neon Tint
                    color_tint = NEON_COLORS[key_name]
                    tinted = (fmap[:, :, None] * color_tint).clip(0, 255).astype(np.uint8)
                    visual_bgr = cv2.resize(tinted, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                elif theme == "Inferno Fire":
                    res = cv2.resize(uint8_f, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                    visual_bgr = cv2.applyColorMap(res, cv2.COLORMAP_INFERNO)
                elif theme == "Plasma":
                    res = cv2.resize(uint8_f, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                    visual_bgr = cv2.applyColorMap(res, cv2.COLORMAP_PLASMA)
                elif theme == "Turbo Spectrum":
                    res = cv2.resize(uint8_f, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                    visual_bgr = cv2.applyColorMap(res, cv2.COLORMAP_TURBO)
                else:  # Monochrome Noir
                    res = cv2.resize(uint8_f, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                    visual_bgr = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

                # Channel Header Banner
                cv2.rectangle(visual_bgr, (0, 0), (tile_w, 26), (15, 15, 20), -1)
                cv2.putText(
                    visual_bgr,
                    label,
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    banner_col,
                    1,
                    cv2.LINE_AA,
                )
                tiles.append(visual_bgr)

            content = np.hstack(tiles)

        else:
            # Cybernetic Fusion View (Composite Tri-Channel Sensory Overlay)
            fusion_h, fusion_w = 480, 854
            # Red = Motion, Green = Gradient Edges (H + V), Blue = Contrast + Brightness
            r_chan = np.clip(f_mot * 1.5, 0.0, 1.0)
            g_chan = np.clip((f_horiz + f_vert) * 0.9, 0.0, 1.0)
            b_chan = np.clip(f_cont * 0.8 + f_bright * 0.3, 0.0, 1.0)

            fusion_rgb = np.stack([b_chan, g_chan, r_chan], axis=2)  # BGR order
            fusion_uint8 = (fusion_rgb * 255.0).astype(np.uint8)
            content = cv2.resize(fusion_uint8, (1380, 520), interpolation=cv2.INTER_NEAREST)

            # Overlay Fusion Legend
            cv2.rectangle(content, (15, 15), (560, 48), (10, 10, 15), -1)
            cv2.putText(
                content,
                "CYBERNETIC SENSORY FUSION [Red=Motion | Green=Edges | Blue=Contrast]",
                (22, 37),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Bottom Info Status Bar
        status_bar = np.zeros((36, content.shape[1], 3), dtype=np.uint8)
        total_signals = extractor.cols * extractor.rows * 5
        status_text = (
            f"Sensory Grid: {extractor.cols}x{extractor.rows} ({total_signals:,} signals) | "
            f"Theme: {theme} [c] | View: {view_mode.upper()} [v] | "
            f"Sens: {sensitivity:.1f}x [[/]] | Decay: {'ON' if extractor.motion_decay > 0 else 'OFF'} [m] | "
            f"{fps:.1f} FPS"
        )
        cv2.putText(
            status_bar,
            status_text,
            (15, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        final_frame = np.vstack([content, status_bar])

        cv2.imshow("Visual Play - Sensory Engine", final_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("k"), ord(" "), ord("p")):
            toggle_camera()
        elif key == ord("v"):
            view_mode = "fusion" if view_mode == "split" else "split"
        elif key == ord("c"):
            theme_idx = (theme_idx + 1) % len(THEMES)
        elif key in (ord("+"), ord("=")):
            extractor.cols = min(80, extractor.cols + 6)
            extractor.rows = min(48, extractor.rows + 4)
            extractor.reset()
        elif key in (ord("-"), ord("_")):
            extractor.cols = max(10, extractor.cols - 6)
            extractor.rows = max(6, extractor.rows - 4)
            extractor.reset()
        elif key == ord("]"):
            sensitivity = min(3.5, sensitivity + 0.2)
        elif key == ord("["):
            sensitivity = max(0.4, sensitivity - 0.2)
        elif key == ord("m"):
            extractor.motion_decay = 0.0 if extractor.motion_decay > 0.0 else 0.75
        elif key == ord("x"):
            extractor.mirror = not extractor.mirror
            extractor.reset()

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(">> Sensory engine closed.")