"""
UI.py

Comparison cockpit for Visual_Play.

Purpose
-------
Provide one stable experiment surface for comparing:

TOP:    the current algorithmic Visual_Play cortical reconstruction
BOTTOM: a future neuron-chain reconstruction driven by the same webcam frame

The bottom path intentionally displays "not connected" until a genuine neural
pipeline is supplied. It never fabricates neural output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import cv2
import numpy as np

from cortical_system import CorticalSystem


@dataclass
class NeuralFrameResult:
    """Minimal contract expected from the future neuron-chain experiment."""

    reconstruction: np.ndarray
    diagnostics: Dict[str, float]


class NeuralSystem(Protocol):
    """Future neural pathway contract. UI owns presentation, not cognition."""

    def process(self, frame: np.ndarray, learn: bool = True) -> NeuralFrameResult:
        ...

    def reset_activity(self) -> None:
        ...


class VisualComparisonUI:
    """Side-by-side-in-time comparison cockpit: current path above, neural path below."""

    WINDOW = "Visual Play - Current vs Neural Signal Flow"
    WIDTH = 1440
    HEADER_H = 58
    PANEL_H = 330
    FOOTER_H = 126

    def __init__(
        self,
        current_system: Optional[CorticalSystem] = None,
        neural_system: Optional[NeuralSystem] = None,
    ) -> None:
        self.current = current_system or CorticalSystem(cols=64, rows=36)
        self.neural = neural_system

        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_on = False
        self.learning_on = True
        self.show_source = True

        self.last_frame: Optional[np.ndarray] = None
        self.last_current: Optional[Dict[str, Any]] = None
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
    def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        if np.issubdtype(image.dtype, np.floating):
            lo = float(np.min(image))
            hi = float(np.max(image))
            if hi - lo > 1e-8:
                image = (image - lo) / (hi - lo)
            else:
                image = np.zeros_like(image)
            image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _label(canvas: np.ndarray, text: str, y: int, scale: float = 0.52) -> None:
        cv2.putText(
            canvas,
            text,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    def _current_reconstruction(self) -> np.ndarray:
        if self.last_current is None:
            return self._blank(self.PANEL_H - 42, self.WIDTH, 5)

        v1 = self.last_current["v1"]
        recon = np.asarray(v1.reconstruction, dtype=np.float32)

        # Diagnostic color projection only. It does not feed learning.
        luminance = recon[0]
        motion = recon[2]
        edges = recon[3] + recon[4]

        blue = np.clip(luminance * 2.0, 0.0, 1.0)
        green = np.clip(edges * 3.0, 0.0, 1.0)
        red = np.clip(motion * 4.0, 0.0, 1.0)
        rgb = np.stack([blue, green, red], axis=2)
        return self._fit_image(rgb, self.WIDTH, self.PANEL_H - 42)

    def _neural_reconstruction(self) -> np.ndarray:
        body_h = self.PANEL_H - 42
        if self.neural is None:
            panel = self._blank(body_h, self.WIDTH, 5)
            cv2.putText(
                panel,
                "NEURAL CHAIN NOT CONNECTED YET",
                (self.WIDTH // 2 - 245, body_h // 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (120, 170, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                "This panel will only show output produced by the neuron-to-neuron pathway.",
                (self.WIDTH // 2 - 320, body_h // 2 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (165, 165, 175),
                1,
                cv2.LINE_AA,
            )
            return panel

        if self.last_neural is None:
            return self._blank(body_h, self.WIDTH, 5)

        return self._fit_image(
            np.asarray(self.last_neural.reconstruction),
            self.WIDTH,
            body_h,
        )

    def _render_panel(self, title: str, subtitle: str, image: np.ndarray) -> np.ndarray:
        panel = self._blank(self.PANEL_H, self.WIDTH, 9)
        cv2.rectangle(panel, (0, 0), (self.WIDTH - 1, 41), (20, 20, 24), -1)
        cv2.putText(
            panel,
            title,
            (16, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            subtitle,
            (430, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (165, 165, 175),
            1,
            cv2.LINE_AA,
        )
        panel[42:, :] = self._fit_image(image, self.WIDTH, self.PANEL_H - 42)
        cv2.rectangle(panel, (0, 0), (self.WIDTH - 1, self.PANEL_H - 1), (70, 70, 78), 1)
        return panel

    def _source_inset(self, canvas: np.ndarray) -> None:
        if not self.show_source:
            return
        inset_w, inset_h = 256, 144
        x0 = self.WIDTH - inset_w - 16
        y0 = self.HEADER_H + 52

        if self.last_frame is not None and self.camera_on:
            src = cv2.flip(self.last_frame, 1)
            src = cv2.resize(src, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
        else:
            src = self._blank(inset_h, inset_w, 4)
            cv2.putText(src, "CAMERA OFF", (72, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 220), 2)

        canvas[y0:y0 + inset_h, x0:x0 + inset_w] = src
        cv2.rectangle(canvas, (x0, y0), (x0 + inset_w, y0 + inset_h), (220, 220, 220), 1)
        cv2.rectangle(canvas, (x0, y0), (x0 + inset_w, y0 + 24), (18, 18, 22), -1)
        cv2.putText(canvas, "SHARED WEBCAM INPUT", (x0 + 8, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

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
        cv2.putText(header, text, (x1 + 9, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)

    def _render_header(self) -> np.ndarray:
        header = self._blank(self.HEADER_H, self.WIDTH, 13)
        self._button(header, "camera", 14, 198, "CAMERA ON/OFF [k]", self.camera_on)
        self._button(header, "learning", 210, 390, "LEARNING [l]", self.learning_on)
        self._button(header, "reset", 402, 555, "RESET STATE [r]")
        self._button(header, "source", 567, 735, "SOURCE INSET [s]", self.show_source)

        neural_status = "CONNECTED" if self.neural is not None else "NOT BUILT"
        status = f"FPS {self.fps:5.1f}   |   NEURAL PATH: {neural_status}"
        cv2.putText(header, status, (850, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 120), 1, cv2.LINE_AA)
        return header

    def _render_footer(self) -> np.ndarray:
        footer = self._blank(self.FOOTER_H, self.WIDTH, 12)
        cv2.line(footer, (0, 0), (self.WIDTH, 0), (70, 70, 78), 1)

        if self.last_current is not None:
            v1 = self.last_current["v1"]
            assoc = self.last_current["assoc"]
            temp = self.last_current["temporal"]
            pari = self.last_current["parietal"]
            self._label(footer, f"CURRENT  V1 active: {v1.active_count} | novelty: {v1.novelty:.3f} | feedback steps: {v1.feedback_steps} | weight change: {v1.mean_weight_change:.6f}", 28)
            self._label(footer, f"STREAMS  ventral gate: {assoc.mean_ventral_gate:.3f} | dorsal gate: {assoc.mean_dorsal_gate:.3f} | temporal stability: {temp.stability:.3f}", 54)
            self._label(footer, f"MOTION   energy: {pari.motion_energy:.4f} | centroid: ({pari.centroid[0]:.2f}, {pari.centroid[1]:.2f}) | vector: ({pari.motion_vector[0]:+.2f}, {pari.motion_vector[1]:+.2f})", 80)

        if self.neural is None:
            self._label(footer, "NEURAL   awaiting genuine neuron-chain owner; no placeholder cognition is being generated", 108, 0.48)
        elif self.last_neural is not None:
            parts = [f"{key}: {value:.4f}" for key, value in sorted(self.last_neural.diagnostics.items())]
            self._label(footer, "NEURAL   " + " | ".join(parts)[:180], 108, 0.48)

        return footer

    def _compose(self) -> np.ndarray:
        top = self._render_panel(
            "TOP - CURRENT CODED SIGNAL FLOW",
            "Existing Visual_Play V1 diagnostic projection",
            self._current_reconstruction(),
        )
        bottom = self._render_panel(
            "BOTTOM - NEURON CHAIN SIGNAL FLOW",
            "Reserved for locally interacting retinotopic neurons",
            self._neural_reconstruction(),
        )
        canvas = np.vstack([self._render_header(), top, bottom, self._render_footer()])
        self._source_inset(canvas)
        return canvas

    # ------------------------------------------------------------------
    # Interaction / runtime
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        self.current.reset_activity()
        self.current.reset_expectation()
        if self.neural is not None:
            self.neural.reset_activity()
        self.last_current = None
        self.last_neural = None

    def _mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x1, y1, x2, y2) in self.buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if name == "camera":
                    self.toggle_camera()
                elif name == "learning":
                    self.learning_on = not self.learning_on
                elif name == "reset":
                    self.reset_state()
                elif name == "source":
                    self.show_source = not self.show_source
                break

    def _handle_key(self, key: int) -> bool:
        key &= 0xFF
        if key in (ord("q"), 27):
            return False
        if key in (ord("k"), ord("p"), 32):
            self.toggle_camera()
        elif key == ord("l"):
            self.learning_on = not self.learning_on
        elif key == ord("r"):
            self.reset_state()
        elif key == ord("s"):
            self.show_source = not self.show_source
        return True

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.WIDTH, self.HEADER_H + 2 * self.PANEL_H + self.FOOTER_H)
        cv2.setMouseCallback(self.WINDOW, self._mouse)

        if not self.connect_camera():
            print(">> Camera unavailable at startup. UI will remain open with camera OFF.")

        print("Visual_Play comparison UI")
        print("  k / space : connect-disconnect camera")
        print("  l         : freeze-enable learning")
        print("  r         : reset current and neural state")
        print("  s         : toggle shared webcam inset")
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
                        # Both paths receive the exact same captured frame.
                        self.last_current = self.current.process(frame, learn=self.learning_on)
                        if self.neural is not None:
                            self.last_neural = self.neural.process(frame, learn=self.learning_on)
                    else:
                        time.sleep(0.01)

                cv2.imshow(self.WINDOW, self._compose())
                running = self._handle_key(cv2.waitKey(1))
        finally:
            self.disconnect_camera()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    VisualComparisonUI().run()
