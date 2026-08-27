"""
UI.py

Visual_Play experiment cockpit.

TOP:
    Raw webcam plus the five fixed sensory maps produced by vision_features.py.

BOTTOM:
    Reserved for a future neuron-chain pathway that receives the same
    VisionFeatures object. No neural output is fabricated before that pathway exists.

The UI owns capture and presentation only. vision_features.py owns sensory extraction.
Future neural code must own neural state, propagation, plasticity, and reconstruction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import cv2
import numpy as np

from vision_features import VisionFeatureExtractor, VisionFeatures


@dataclass
class NeuralFrameResult:
    """Minimal result contract for the future neuron-chain experiment."""

    reconstruction: np.ndarray
    diagnostics: Dict[str, float]


class NeuralSystem(Protocol):
    """Future neural pathway contract; receives the exact sensory maps shown above."""

    def process(
        self,
        features: VisionFeatures,
        learn: bool = True,
    ) -> NeuralFrameResult:
        ...

    def reset_activity(self) -> None:
        ...


class VisualComparisonUI:
    """Display trustworthy sensory input above and future neural output below."""

    WINDOW = "Visual Play - Sensory Input vs Neural Signal Flow"
    WIDTH = 1440
    HEADER_H = 58
    TOP_H = 440
    BOTTOM_H = 300
    FOOTER_H = 86

    def __init__(
        self,
        extractor: Optional[VisionFeatureExtractor] = None,
        neural_system: Optional[NeuralSystem] = None,
    ) -> None:
        self.extractor = extractor or VisionFeatureExtractor(
            cols=64,
            rows=36,
            mirror=True,
            motion_decay=0.70,
            contrast_gain=6.5,
            motion_gain=7.5,
            edge_gain=4.0,
        )
        self.neural = neural_system

        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_on = False
        self.learning_on = True

        self.last_frame: Optional[np.ndarray] = None
        self.last_features: Optional[VisionFeatures] = None
        self.last_neural: Optional[NeuralFrameResult] = None

        self.fps = 0.0
        self._last_time = time.time()
        self.buttons: Dict[str, tuple[int, int, int, int]] = {}

    # ------------------------------------------------------------------
    # Camera ownership
    # ------------------------------------------------------------------

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        candidates = []
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            candidates.append((0, cv2.CAP_AVFOUNDATION))
        candidates.append((0, cv2.CAP_ANY))

        for index, backend in candidates:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            cap.release()
        return None

    def connect_camera(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.cap = self._open_camera()
        self.camera_on = bool(self.cap is not None and self.cap.isOpened())
        return self.camera_on

    def disconnect_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.camera_on = False

    def toggle_camera(self) -> None:
        if self.camera_on:
            self.disconnect_camera()
            print(">> CAMERA PHYSICALLY DISCONNECTED")
        elif self.connect_camera():
            print(">> CAMERA CONNECTED")
        else:
            print(">> CAMERA CONNECTION FAILED")

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _blank(height: int, width: int, value: int = 8) -> np.ndarray:
        return np.full((height, width, 3), value, dtype=np.uint8)

    @staticmethod
    def _to_bgr_unit_map(feature_map: np.ndarray) -> np.ndarray:
        """Render a true 0..1 feature map without per-frame contrast rescaling."""
        unit = np.nan_to_num(
            np.asarray(feature_map, dtype=np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        unit = np.clip(unit, 0.0, 1.0)
        gray = (unit * 255.0).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _fit_image(
        image: np.ndarray,
        width: int,
        height: int,
        interpolation: int = cv2.INTER_NEAREST,
    ) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return cv2.resize(image, (width, height), interpolation=interpolation)

    @staticmethod
    def _label(
        canvas: np.ndarray,
        text: str,
        y: int,
        scale: float = 0.50,
    ) -> None:
        cv2.putText(
            canvas,
            text,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (225, 225, 230),
            1,
            cv2.LINE_AA,
        )

    def _tile(
        self,
        image: np.ndarray,
        title: str,
        subtitle: str,
        width: int,
        height: int,
        *,
        smooth: bool = False,
    ) -> np.ndarray:
        tile = self._blank(height, width, 9)
        body_h = height - 45
        fitted = self._fit_image(
            image,
            width,
            body_h,
            cv2.INTER_AREA if smooth else cv2.INTER_NEAREST,
        )
        tile[45:, :] = fitted
        cv2.rectangle(tile, (0, 0), (width - 1, 44), (20, 20, 24), -1)
        cv2.putText(
            tile,
            title,
            (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            subtitle,
            (10, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (160, 165, 175),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (62, 62, 70), 1)
        return tile

    def _camera_image(self) -> np.ndarray:
        if self.last_frame is None or not self.camera_on:
            image = self._blank(180, 320, 4)
            cv2.putText(
                image,
                "CAMERA OFF",
                (92, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 220),
                2,
                cv2.LINE_AA,
            )
            return image

        # VisionFeatureExtractor mirrors internally; mirror source display to match it.
        return cv2.flip(self.last_frame, 1)

    def _render_sensory_panel(self) -> np.ndarray:
        panel = self._blank(self.TOP_H, self.WIDTH, 7)

        title_h = 42
        cv2.rectangle(panel, (0, 0), (self.WIDTH - 1, title_h - 1), (16, 16, 20), -1)
        cv2.putText(
            panel,
            "TOP - TRUSTWORTHY SENSORY INPUT",
            (16, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "webcam + fixed vision_features.py maps; no cortical reconstruction",
            (390, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (165, 170, 180),
            1,
            cv2.LINE_AA,
        )

        gap = 8
        cols = 3
        rows = 2
        tile_w = (self.WIDTH - gap * (cols + 1)) // cols
        tile_h = (self.TOP_H - title_h - gap * (rows + 1)) // rows

        feature_items = []
        if self.last_features is not None:
            f = self.last_features
            feature_items = [
                ("BRIGHTNESS", f.brightness),
                ("CONTRAST", f.contrast),
                ("MOTION", f.motion),
                ("HORIZONTAL EDGES", f.horizontal),
                ("VERTICAL EDGES", f.vertical),
            ]

        tiles = [
            self._tile(
                self._camera_image(),
                "WEBCAM",
                "shared physical input",
                tile_w,
                tile_h,
                smooth=True,
            )
        ]

        for title, fmap in feature_items:
            mean = float(np.mean(fmap))
            peak = float(np.max(fmap))
            tiles.append(
                self._tile(
                    self._to_bgr_unit_map(fmap),
                    title,
                    f"mean {mean:.3f} | peak {peak:.3f}",
                    tile_w,
                    tile_h,
                )
            )

        while len(tiles) < 6:
            tiles.append(
                self._tile(
                    self._blank(100, 100, 4),
                    "WAITING",
                    "no sensory frame yet",
                    tile_w,
                    tile_h,
                )
            )

        for index, tile in enumerate(tiles[:6]):
            r = index // cols
            c = index % cols
            x = gap + c * (tile_w + gap)
            y = title_h + gap + r * (tile_h + gap)
            panel[y:y + tile_h, x:x + tile_w] = tile

        return panel

    def _render_neural_panel(self) -> np.ndarray:
        panel = self._blank(self.BOTTOM_H, self.WIDTH, 6)
        cv2.rectangle(panel, (0, 0), (self.WIDTH - 1, 41), (16, 16, 20), -1)
        cv2.putText(
            panel,
            "BOTTOM - NEURON CHAIN SIGNAL FLOW",
            (16, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "future locally interacting neural fields receive the exact maps shown above",
            (440, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (165, 170, 180),
            1,
            cv2.LINE_AA,
        )

        body_h = self.BOTTOM_H - 42

        if self.neural is None:
            cv2.putText(
                panel,
                "NEURAL CHAIN NOT CONNECTED YET",
                (self.WIDTH // 2 - 245, 42 + body_h // 2 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (120, 170, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                "No placeholder cognition or fake reconstruction is being generated.",
                (self.WIDTH // 2 - 300, 42 + body_h // 2 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (165, 165, 175),
                1,
                cv2.LINE_AA,
            )
            return panel

        if self.last_neural is not None:
            recon = np.asarray(self.last_neural.reconstruction)
            if recon.ndim == 2:
                recon = self._to_bgr_unit_map(recon)
            elif np.issubdtype(recon.dtype, np.floating):
                recon = np.clip(np.nan_to_num(recon), 0.0, 1.0)
                recon = (recon * 255.0).astype(np.uint8)
            else:
                recon = np.clip(recon, 0, 255).astype(np.uint8)

            panel[42:, :] = self._fit_image(
                recon,
                self.WIDTH,
                body_h,
                cv2.INTER_NEAREST,
            )

        return panel

    def _button(
        self,
        header: np.ndarray,
        name: str,
        x1: int,
        x2: int,
        text: str,
        active: bool = False,
    ) -> None:
        y1, y2 = 10, 47
        self.buttons[name] = (x1, y1, x2, y2)
        fill = (55, 95, 55) if active else (42, 42, 48)
        border = (110, 215, 120) if active else (105, 105, 120)
        cv2.rectangle(header, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(header, (x1, y1), (x2, y2), border, 1)
        cv2.putText(
            header,
            text,
            (x1 + 9, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    def _render_header(self) -> np.ndarray:
        header = self._blank(self.HEADER_H, self.WIDTH, 13)
        self._button(
            header,
            "camera",
            14,
            198,
            "CAMERA ON/OFF [k]",
            self.camera_on,
        )
        self._button(
            header,
            "reset",
            210,
            380,
            "RESET SENSORY [r]",
        )
        self._button(
            header,
            "learning",
            392,
            590,
            "NEURAL LEARNING [l]",
            self.learning_on,
        )

        neural_status = "CONNECTED" if self.neural is not None else "NOT BUILT"
        status = f"FPS {self.fps:5.1f}   |   NEURAL PATH: {neural_status}"
        cv2.putText(
            header,
            status,
            (850, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 120),
            1,
            cv2.LINE_AA,
        )
        return header

    def _render_footer(self) -> np.ndarray:
        footer = self._blank(self.FOOTER_H, self.WIDTH, 12)
        cv2.line(footer, (0, 0), (self.WIDTH, 0), (70, 70, 78), 1)

        if self.last_features is not None:
            f = self.last_features
            self._label(
                footer,
                "SENSORY  "
                f"brightness {np.mean(f.brightness):.3f} | "
                f"contrast {np.mean(f.contrast):.3f} | "
                f"motion {np.mean(f.motion):.3f} | "
                f"H-edge {np.mean(f.horizontal):.3f} | "
                f"V-edge {np.mean(f.vertical):.3f}",
                28,
            )

        if self.neural is None:
            self._label(
                footer,
                "NEURAL   awaiting genuine neuron-chain owner; top panel is only the measured sensory baseline",
                58,
                0.48,
            )
        elif self.last_neural is not None:
            parts = [
                f"{key}: {value:.4f}"
                for key, value in sorted(self.last_neural.diagnostics.items())
            ]
            self._label(
                footer,
                "NEURAL   " + " | ".join(parts)[:180],
                58,
                0.48,
            )

        return footer

    def _compose(self) -> np.ndarray:
        return np.vstack(
            [
                self._render_header(),
                self._render_sensory_panel(),
                self._render_neural_panel(),
                self._render_footer(),
            ]
        )

    # ------------------------------------------------------------------
    # Interaction / runtime
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        self.extractor.reset()
        if self.neural is not None:
            self.neural.reset_activity()
        self.last_features = None
        self.last_neural = None

    def _mouse(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: Any,
    ) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for name, (x1, y1, x2, y2) in self.buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if name == "camera":
                    self.toggle_camera()
                elif name == "reset":
                    self.reset_state()
                elif name == "learning":
                    self.learning_on = not self.learning_on
                break

    def _handle_key(self, key: int) -> bool:
        key &= 0xFF
        if key in (ord("q"), 27):
            return False
        if key in (ord("k"), ord("p"), 32):
            self.toggle_camera()
        elif key == ord("r"):
            self.reset_state()
        elif key == ord("l"):
            self.learning_on = not self.learning_on
        return True

    def run(self) -> None:
        total_h = self.HEADER_H + self.TOP_H + self.BOTTOM_H + self.FOOTER_H
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.WIDTH, total_h)
        cv2.setMouseCallback(self.WINDOW, self._mouse)

        if not self.connect_camera():
            print(">> Camera unavailable at startup. UI will remain open with camera OFF.")

        print("Visual_Play sensory-vs-neural UI")
        print("  k / space : connect-disconnect camera")
        print("  r         : reset sensory temporal state")
        print("  l         : freeze-enable future neural learning")
        print("  q / esc   : quit")

        try:
            running = True
            while running:
                now = time.time()
                dt = max(now - self._last_time, 1e-6)
                self._last_time = now
                self.fps = 0.90 * self.fps + 0.10 * (1.0 / dt)

                if self.camera_on and self.cap is not None:
                    ok, frame = self.cap.read()
                    if ok:
                        self.last_frame = frame.copy()

                        # Extract once. The top displays these exact maps and the
                        # future neural path receives the same VisionFeatures object.
                        self.last_features = self.extractor.extract(frame)

                        if self.neural is not None:
                            self.last_neural = self.neural.process(
                                self.last_features,
                                learn=self.learning_on,
                            )
                    else:
                        time.sleep(0.01)

                cv2.imshow(self.WINDOW, self._compose())
                running = self._handle_key(cv2.waitKey(1))
        finally:
            self.disconnect_camera()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    VisualComparisonUI().run()
