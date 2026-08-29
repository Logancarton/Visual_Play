"""
UI.py

Visual_Play experiment-builder cockpit.

The UI owns camera capture, presentation, experiment editing, notes, graph
layout visualization, and observation surfaces. It does NOT own neural
propagation, membrane state, plasticity math, or image reconstruction.

A future neural engine consumes ExperimentSpec and may return retinotopic layer
views, connection strengths, and a traceable visual projection. The UI will
show those outputs without arbitrarily reshaping vectors into pictures.
"""

from __future__ import annotations

import math
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import cv2
import numpy as np

from experiment_spec import (
    CONNECTION_PATTERNS,
    MECHANISMS,
    MODULATOR_PROFILES,
    PLASTICITY_RULES,
    SENSORY_LABELS,
    SENSORY_SOURCES,
    SIGNAL_PROFILES,
    VISUALIZATION_MODES,
    ConnectionSpec,
    ExperimentSpec,
    LayerSpec,
)
from graph_layout import apply_grid, apply_relaxation
from vision_features import VisionFeatureExtractor, VisionFeatures


EXPERIMENT_FILE = Path("experiment.local.json")


@dataclass
class NeuralFrameResult:
    """Future engine result contract; no neural owner exists in the UI."""

    layer_views: Dict[str, Dict[str, np.ndarray]]
    diagnostics: Dict[str, float]
    connection_strengths: Dict[str, float] = field(default_factory=dict)
    visual_projection: Optional[np.ndarray] = None
    projection_provenance: str = ""


class NeuralEngine(Protocol):
    def configure(self, spec: ExperimentSpec) -> None:
        ...

    def process(
        self,
        features: VisionFeatures,
        *,
        learn: bool = True,
    ) -> NeuralFrameResult:
        ...

    def reset_activity(self) -> None:
        ...


class VisualExperimentUI:
    WINDOW = "Visual Play - Neural Experiment Laboratory"

    WIDTH = 1500
    HEADER_H = 58
    BUILDER_H = 480
    OBSERVATION_H = 300
    FOOTER_H = 62

    LEFT_W = 230
    RIGHT_W = 420
    CENTER_W = WIDTH - LEFT_W - RIGHT_W

    NODE_W = 172
    NODE_H = 80
    SENSOR_W = 118
    SENSOR_H = 50

    def __init__(
        self,
        extractor: Optional[VisionFeatureExtractor] = None,
        engine: Optional[NeuralEngine] = None,
        spec: Optional[ExperimentSpec] = None,
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
        self.engine = engine
        self.spec = spec or ExperimentSpec.default()

        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_on = False
        self.capture_paused = False
        self.learning_on = True

        self.last_frame: Optional[np.ndarray] = None
        self.last_features: Optional[VisionFeatures] = None
        self.last_neural: Optional[NeuralFrameResult] = None

        self.selected_kind: Optional[str] = "layer"
        self.selected_id: Optional[str] = self.spec.layers[0].id if self.spec.layers else None

        self.connect_mode = False
        self.pending_source: Optional[str] = None
        self.dragging_layer_id: Optional[str] = None
        self.drag_offset = (0, 0)

        self.controls: Dict[str, Tuple[int, int, int, int]] = {}
        self.node_rects: Dict[str, Tuple[int, int, int, int]] = {}
        self.connection_segments: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {}
        self.connection_flash: Dict[str, float] = {}

        self.note_edit_target: Optional[Tuple[str, Optional[str]]] = None
        self.note_buffer = ""

        self.fps = 0.0
        self._last_time = time.time()
        self._last_layout_time = 0.0
        self.status_message = (
            "Builder ready. Neural mechanisms are configuration-only until a separate engine exists."
        )
        self._configure_engine()

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
            self.status_message = "Camera physically disconnected."
        elif self.connect_camera():
            self.status_message = "Camera connected."
        else:
            self.status_message = "Camera connection failed."

    # ------------------------------------------------------------------
    # Architecture boundaries
    # ------------------------------------------------------------------

    def _configure_engine(self) -> None:
        if self.engine is not None:
            self.engine.configure(self.spec)

    def _spec_changed(self, message: str) -> None:
        self.spec.validate()
        self._configure_engine()
        self.last_neural = None
        self.status_message = message

    def _runtime_strengths(self) -> Dict[str, float]:
        if self.last_neural is None:
            return {}
        return dict(self.last_neural.connection_strengths)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _blank(height: int, width: int, value: int = 10) -> np.ndarray:
        return np.full((height, width, 3), value, dtype=np.uint8)

    @staticmethod
    def _text(
        canvas: np.ndarray,
        text: str,
        x: int,
        y: int,
        scale: float = 0.45,
        color: Tuple[int, int, int] = (225, 225, 230),
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            canvas,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _wrapped_text(
        self,
        canvas: np.ndarray,
        text: str,
        x: int,
        y: int,
        *,
        width_chars: int,
        max_lines: int,
        line_height: int = 19,
        scale: float = 0.31,
        color: Tuple[int, int, int] = (165, 170, 180),
    ) -> None:
        cleaned = " ".join(str(text).split())
        lines = textwrap.wrap(cleaned, width=max(10, width_chars)) or [""]
        for index, line in enumerate(lines[:max_lines]):
            suffix = "..." if index == max_lines - 1 and len(lines) > max_lines else ""
            self._text(canvas, line + suffix, x, y + index * line_height, scale, color)

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
    def _to_bgr_unit_map(feature_map: np.ndarray) -> np.ndarray:
        unit = np.nan_to_num(
            np.asarray(feature_map, dtype=np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        gray = (np.clip(unit, 0.0, 1.0) * 255.0).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _runtime_image(self, value: np.ndarray) -> Optional[np.ndarray]:
        array = np.asarray(value)
        if array.ndim == 2:
            return self._to_bgr_unit_map(array)
        if array.ndim == 3 and array.shape[2] == 3:
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(np.nan_to_num(array), 0.0, 1.0)
                return (array * 255.0).astype(np.uint8)
            return np.clip(array, 0, 255).astype(np.uint8)
        return None

    def _button(
        self,
        canvas: np.ndarray,
        key: str,
        rect: Tuple[int, int, int, int],
        label: str,
        *,
        active: bool = False,
        enabled: bool = True,
        scale: float = 0.40,
    ) -> None:
        x1, y1, x2, y2 = rect
        self.controls[key] = rect
        if not enabled:
            fill, border, text = (28, 28, 32), (55, 55, 62), (100, 100, 108)
        elif active:
            fill, border, text = (55, 85, 62), (110, 210, 130), (245, 245, 245)
        else:
            fill, border, text = (38, 38, 44), (90, 90, 104), (228, 228, 232)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)
        self._text(canvas, label, x1 + 8, y1 + (y2 - y1) // 2 + 5, scale, text)

    def _panel_title(
        self,
        canvas: np.ndarray,
        title: str,
        subtitle: str,
        x: int,
        y: int,
        width: int,
    ) -> None:
        cv2.rectangle(canvas, (x, y), (x + width - 1, y + 38), (19, 19, 24), -1)
        self._text(canvas, title, x + 12, y + 24, 0.52, (245, 245, 245))
        if subtitle:
            self._text(canvas, subtitle, x + 12, y + 36, 0.28, (145, 150, 162))

    # ------------------------------------------------------------------
    # Header / tools
    # ------------------------------------------------------------------

    def _render_header(self) -> np.ndarray:
        header = self._blank(self.HEADER_H, self.WIDTH, 13)
        self._button(header, "camera", (12, 10, 174, 47), "CAMERA [k]", active=self.camera_on)
        self._button(
            header,
            "pause",
            (184, 10, 334, 47),
            "PAUSE [p]" if not self.capture_paused else "RESUME [p]",
            active=self.capture_paused,
        )
        self._button(header, "reset_sensory", (344, 10, 484, 47), "RESET INPUT")
        self._button(
            header,
            "learning",
            (494, 10, 650, 47),
            "PLASTICITY [l]",
            active=self.learning_on and self.engine is not None,
            enabled=self.engine is not None,
        )
        self._button(header, "save", (660, 10, 770, 47), "SAVE [s]")
        self._button(header, "load", (780, 10, 890, 47), "LOAD [o]")

        engine_status = "CONNECTED" if self.engine is not None else "NOT BUILT"
        status = (
            f"FPS {self.fps:5.1f} | layers {len(self.spec.layers)} | links {len(self.spec.connections)} | "
            f"layout {self.spec.layout_mode.upper()} | engine {engine_status}"
        )
        self._text(header, status, 930, 34, 0.42, (218, 218, 130))
        return header

    def _render_tools(self, builder: np.ndarray) -> None:
        self._panel_title(
            builder,
            "EXPERIMENT",
            "structure and intent; neural math stays separate",
            0,
            0,
            self.LEFT_W,
        )
        y = 48
        actions = [
            ("add_layer", "+ ADD LAYER [a]"),
            ("add_branch", "+ BRANCH [b]"),
            ("connect", "CONNECT NODES [c]"),
            ("delete", "DELETE SELECTED"),
            ("layout_toggle", f"AUTO PULL: {self.spec.layout_mode.upper()} [g]"),
            ("reset_grid", "RESET GRID [r]"),
            ("experiment_notes", "EXPERIMENT NOTES [e]"),
        ]
        for key, label in actions:
            self._button(
                builder,
                key,
                (12, y, self.LEFT_W - 12, y + 31),
                label,
                active=(key == "connect" and self.connect_mode)
                or (key == "layout_toggle" and self.spec.layout_mode == "dynamic")
                or (key == "experiment_notes" and self.selected_kind == "experiment"),
                scale=0.34,
            )
            y += 36

        self._text(builder, "ENGINE VIEW CONFIG", 12, y + 9, 0.34, (170, 175, 185))
        y += 18
        col_w = 98
        for index, mode in enumerate(VISUALIZATION_MODES):
            row, col = divmod(index, 2)
            x1 = 12 + col * (col_w + 8)
            y1 = y + row * 31
            self._button(
                builder,
                f"view:{mode}",
                (x1, y1, x1 + col_w, y1 + 25),
                mode.upper(),
                active=self.spec.visualization_mode == mode,
                scale=0.27,
            )

        if self.connect_mode:
            source = self._display_name(self.pending_source) if self.pending_source else "click source"
            self._text(builder, "CONNECT MODE", 12, self.BUILDER_H - 35, 0.37, (120, 205, 255))
            self._text(builder, f"{source} -> target", 12, self.BUILDER_H - 15, 0.29)

    # ------------------------------------------------------------------
    # Network graph
    # ------------------------------------------------------------------

    def _sensor_center(self, sensor_id: str) -> Tuple[int, int]:
        index = SENSORY_SOURCES.index(sensor_id)
        gap = 10
        total = len(SENSORY_SOURCES) * self.SENSOR_W + (len(SENSORY_SOURCES) - 1) * gap
        start_x = self.LEFT_W + (self.CENTER_W - total) // 2
        x1 = start_x + index * (self.SENSOR_W + gap)
        y1 = 54
        return (x1 + self.SENSOR_W // 2, y1 + self.SENSOR_H // 2)

    def _layer_rect(self, layer: LayerSpec) -> Tuple[int, int, int, int]:
        graph_x1 = self.LEFT_W + 10
        graph_y1 = 118
        graph_w = self.CENTER_W - 20
        graph_h = self.BUILDER_H - 132
        cx = graph_x1 + int(layer.x * graph_w)
        cy = graph_y1 + int(layer.y * graph_h)
        x1 = int(np.clip(cx - self.NODE_W // 2, graph_x1, graph_x1 + graph_w - self.NODE_W))
        y1 = int(np.clip(cy - self.NODE_H // 2, graph_y1, graph_y1 + graph_h - self.NODE_H))
        return (x1, y1, x1 + self.NODE_W, y1 + self.NODE_H)

    def _display_name(self, item_id: Optional[str]) -> str:
        if item_id is None:
            return "none"
        if item_id in SENSORY_LABELS:
            return SENSORY_LABELS[item_id]
        layer = self.spec.layer_by_id(item_id)
        if layer is not None:
            return layer.name
        connection = self.spec.connection_by_id(item_id)
        if connection is not None:
            return f"{self._display_name(connection.source_id)} -> {self._display_name(connection.target_id)}"
        return item_id

    def _node_center(self, node_id: str) -> Optional[Tuple[int, int]]:
        if node_id in SENSORY_SOURCES:
            return self._sensor_center(node_id)
        rect = self.node_rects.get(node_id)
        if rect is None:
            layer = self.spec.layer_by_id(node_id)
            if layer is None:
                return None
            rect = self._layer_rect(layer)
        return ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)

    def _connection_strength(self, connection: ConnectionSpec) -> float:
        runtime = self._runtime_strengths()
        return float(runtime.get(connection.id, connection.gain))

    def _draw_arrow(
        self,
        canvas: np.ndarray,
        connection: ConnectionSpec,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        *,
        selected: bool,
    ) -> None:
        now = time.monotonic()
        flash_age = now - self.connection_flash.get(connection.id, -999.0)
        flashing = 0.0 <= flash_age <= 1.4
        strength = max(0.0, self._connection_strength(connection))

        if flashing:
            color, thickness = (100, 220, 255), 4
        elif selected:
            color, thickness = (110, 220, 255), 3
        elif connection.signal == "inhibitory":
            color, thickness = (175, 120, 220), 2
        elif connection.signal == "mixed":
            color, thickness = (110, 180, 200), 2
        else:
            color, thickness = (110, 165, 125), 2

        thickness = max(thickness, min(5, 1 + int(round(strength))))
        cv2.arrowedLine(canvas, p1, p2, color, thickness, cv2.LINE_AA, tipLength=0.045)

        if flashing:
            phase = min(1.0, flash_age / 1.2)
            px = int(p1[0] + (p2[0] - p1[0]) * phase)
            py = int(p1[1] + (p2[1] - p1[1]) * phase)
            cv2.circle(canvas, (px, py), 7, (245, 245, 245), -1, cv2.LINE_AA)
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            self._text(canvas, "NEW CONNECTION", mx - 55, my - 10, 0.28, (160, 225, 255))

    def _render_graph(self, builder: np.ndarray) -> None:
        x0 = self.LEFT_W
        self._panel_title(
            builder,
            "NETWORK / SIGNAL GRAPH",
            "branches, loops, saved positions, and dynamic connection pull",
            x0,
            0,
            self.CENTER_W,
        )
        self.node_rects.clear()
        self.connection_segments.clear()

        for sensor_id in SENSORY_SOURCES:
            cx, cy = self._sensor_center(sensor_id)
            rect = (
                cx - self.SENSOR_W // 2,
                cy - self.SENSOR_H // 2,
                cx + self.SENSOR_W // 2,
                cy + self.SENSOR_H // 2,
            )
            self.node_rects[sensor_id] = rect
            selected = self.selected_kind == "sensor" and self.selected_id == sensor_id
            fill = (42, 48, 54) if not selected else (55, 75, 86)
            border = (90, 105, 118) if not selected else (110, 210, 240)
            cv2.rectangle(builder, (rect[0], rect[1]), (rect[2], rect[3]), fill, -1)
            cv2.rectangle(builder, (rect[0], rect[1]), (rect[2], rect[3]), border, 2 if selected else 1)
            self._text(builder, SENSORY_LABELS[sensor_id].upper(), rect[0] + 8, rect[1] + 22, 0.33)
            tag = "DIRECT LIGHT" if sensor_id == "sensor:brightness" else "DIAGNOSTIC"
            self._text(builder, tag, rect[0] + 8, rect[1] + 40, 0.24, (145, 150, 160))

        for layer in self.spec.layers:
            self.node_rects[layer.id] = self._layer_rect(layer)

        for connection in self.spec.connections:
            p1 = self._node_center(connection.source_id)
            p2 = self._node_center(connection.target_id)
            if p1 is None or p2 is None:
                continue
            tx, ty = p2
            sx, sy = p1
            dx, dy = tx - sx, ty - sy
            length = max(math.hypot(dx, dy), 1.0)
            p2_draw = (
                int(tx - dx / length * (self.NODE_W * 0.34)),
                int(ty - dy / length * (self.NODE_H * 0.34)),
            )
            selected = self.selected_kind == "connection" and self.selected_id == connection.id
            self._draw_arrow(builder, connection, p1, p2_draw, selected=selected)
            self.connection_segments[connection.id] = (p1, p2_draw)

        for layer in self.spec.layers:
            rect = self.node_rects[layer.id]
            selected = self.selected_kind == "layer" and self.selected_id == layer.id
            fill = (38, 43, 48) if not selected else (50, 70, 60)
            border = (85, 95, 106) if not selected else (115, 220, 140)
            cv2.rectangle(builder, (rect[0], rect[1]), (rect[2], rect[3]), fill, -1)
            cv2.rectangle(builder, (rect[0], rect[1]), (rect[2], rect[3]), border, 2 if selected else 1)
            self._text(builder, layer.name, rect[0] + 10, rect[1] + 21, 0.46, (245, 245, 245))
            self._text(
                builder,
                f"{layer.rows} x {layer.cols} = {layer.unit_count:,}",
                rect[0] + 10,
                rect[1] + 43,
                0.31,
                (175, 180, 188),
            )
            enabled = sum(1 for value in layer.mechanisms.values() if value)
            pin = "PIN" if layer.pinned else "AUTO"
            self._text(
                builder,
                f"{enabled} mechanisms | {pin} | {self.spec.visualization_mode}",
                rect[0] + 10,
                rect[1] + 63,
                0.27,
                (140, 160, 150),
            )

        if self.engine is None:
            self._text(
                builder,
                "GRAPH IS REAL SPECIFICATION; NEURAL ACTIVITY IS NOT SIMULATED",
                x0 + 14,
                self.BUILDER_H - 12,
                0.31,
                (130, 135, 145),
            )

    # ------------------------------------------------------------------
    # Inspector and notes
    # ------------------------------------------------------------------

    def _render_inspector(self, builder: np.ndarray) -> None:
        x0 = self.LEFT_W + self.CENTER_W
        self._panel_title(builder, "INSPECTOR", "selected object + intent", x0, 0, self.RIGHT_W)

        if self.selected_kind == "experiment":
            self._render_experiment_inspector(builder, x0)
            return
        if self.selected_kind == "layer" and self.selected_id:
            layer = self.spec.layer_by_id(self.selected_id)
            if layer is not None:
                self._render_layer_inspector(builder, x0, layer)
                return
        if self.selected_kind == "connection" and self.selected_id:
            connection = self.spec.connection_by_id(self.selected_id)
            if connection is not None:
                self._render_connection_inspector(builder, x0, connection)
                return
        if self.selected_kind == "sensor" and self.selected_id in SENSORY_SOURCES:
            self._render_sensor_inspector(builder, x0, self.selected_id)
            return

        self._text(builder, "Nothing selected.", x0 + 16, 76, 0.44)

    def _render_experiment_inspector(self, builder: np.ndarray, x0: int) -> None:
        self._text(builder, self.spec.name.upper(), x0 + 14, 72, 0.47, (245, 245, 245))
        self._text(
            builder,
            f"version {self.spec.version} | layout {self.spec.layout_mode} | "
            f"{len(self.spec.layers)} fields | {len(self.spec.connections)} links",
            x0 + 14,
            98,
            0.32,
            (165, 170, 180),
        )
        self._text(builder, "LOGIC / EXPERIMENT NOTES", x0 + 14, 132, 0.36, (185, 190, 200))
        self._wrapped_text(builder, self.spec.notes, x0 + 14, 156, width_chars=55, max_lines=10)
        self._button(builder, "edit_notes", (x0 + 14, 400, x0 + 394, 435), "EDIT NOTES [n]")
        self._text(
            builder,
            "Notes are saved in experiment.local.json and do not alter behavior.",
            x0 + 14,
            460,
            0.27,
            (135, 140, 150),
        )

    def _dimension_row(self, builder, x0, y, label, value, prefix) -> None:
        self._text(builder, label, x0 + 14, y + 20, 0.35, (175, 180, 190))
        self._button(builder, f"{prefix}:-10", (x0 + 86, y, x0 + 126, y + 27), "-10", scale=0.27)
        self._button(builder, f"{prefix}:-1", (x0 + 131, y, x0 + 166, y + 27), "-1", scale=0.27)
        cv2.rectangle(builder, (x0 + 172, y), (x0 + 254, y + 27), (26, 26, 31), -1)
        self._text(builder, str(value), x0 + 190, y + 19, 0.37)
        self._button(builder, f"{prefix}:+1", (x0 + 260, y, x0 + 295, y + 27), "+1", scale=0.27)
        self._button(builder, f"{prefix}:+10", (x0 + 300, y, x0 + 344, y + 27), "+10", scale=0.27)

    def _render_layer_inspector(self, builder: np.ndarray, x0: int, layer: LayerSpec) -> None:
        self._text(builder, layer.name.upper(), x0 + 14, 68, 0.48, (245, 245, 245))
        self._text(
            builder,
            f"{layer.unit_count:,} neurons | position ({layer.x:.2f}, {layer.y:.2f})",
            x0 + 14,
            90,
            0.30,
            (155, 160, 170),
        )
        self._dimension_row(builder, x0, 102, "ROWS", layer.rows, "rows")
        self._dimension_row(builder, x0, 134, "COLS", layer.cols, "cols")
        self._button(
            builder,
            "pin_layer",
            (x0 + 14, 170, x0 + 394, 198),
            "PINNED - MANUAL POSITION" if layer.pinned else "AUTO - MOVES WITH CONNECTIONS",
            active=layer.pinned,
            scale=0.32,
        )
        self._text(builder, "CONFIGURED MECHANISMS - NOT SIMULATED", x0 + 14, 220, 0.33, (180, 185, 195))
        start_y = 231
        col_w = 184
        for index, mechanism in enumerate(MECHANISMS):
            row, col = divmod(index, 2)
            x1 = x0 + 14 + col * (col_w + 8)
            y1 = start_y + row * 31
            self._button(
                builder,
                f"mech:{mechanism}",
                (x1, y1, x1 + col_w, y1 + 25),
                mechanism.replace("_", " ").upper(),
                active=layer.mechanisms.get(mechanism, False),
                scale=0.26,
            )
        self._text(builder, "NOTES", x0 + 14, 424, 0.31, (180, 185, 195))
        self._wrapped_text(builder, layer.notes, x0 + 64, 424, width_chars=42, max_lines=2, line_height=17, scale=0.27)
        self._button(builder, "edit_notes", (x0 + 14, 449, x0 + 394, 474), "EDIT ITEM NOTES [n]", scale=0.31)

    def _cycle_row(self, builder, x0, y, label, value, key) -> None:
        self._text(builder, label, x0 + 14, y + 20, 0.34, (175, 180, 190))
        self._button(
            builder,
            key,
            (x0 + 126, y, x0 + 394, y + 27),
            value.replace("_", " ").upper(),
            scale=0.30,
        )

    def _number_row(self, builder, x0, y, label, value_text, prefix) -> None:
        self._text(builder, label, x0 + 14, y + 20, 0.34, (175, 180, 190))
        self._button(builder, f"{prefix}:-", (x0 + 126, y, x0 + 164, y + 27), "-")
        cv2.rectangle(builder, (x0 + 170, y), (x0 + 340, y + 27), (26, 26, 31), -1)
        self._text(builder, value_text, x0 + 184, y + 19, 0.36)
        self._button(builder, f"{prefix}:+", (x0 + 346, y, x0 + 394, y + 27), "+")

    def _render_connection_inspector(self, builder, x0, connection: ConnectionSpec) -> None:
        self._text(builder, "CONNECTION", x0 + 14, 68, 0.48, (245, 245, 245))
        self._text(
            builder,
            f"{self._display_name(connection.source_id)} -> {self._display_name(connection.target_id)}",
            x0 + 14,
            91,
            0.32,
            (175, 180, 190),
        )
        runtime = self._runtime_strengths().get(connection.id)
        strength_label = f"runtime {runtime:.3f}" if runtime is not None else f"configured gain {connection.gain:.2f}"
        self._text(builder, strength_label, x0 + 14, 111, 0.29, (150, 165, 155))

        self._cycle_row(builder, x0, 124, "PATTERN", connection.pattern, "conn:pattern")
        self._cycle_row(builder, x0, 157, "SIGNAL", connection.signal, "conn:signal")
        self._cycle_row(builder, x0, 190, "PLASTICITY", connection.plasticity, "conn:plasticity")
        self._cycle_row(builder, x0, 223, "MODULATOR", connection.modulator, "conn:modulator")
        self._number_row(builder, x0, 263, "RADIUS", str(connection.radius), "conn:radius")
        self._number_row(builder, x0, 296, "DENSITY", f"{connection.density * 100:.0f}%", "conn:density")
        self._number_row(builder, x0, 329, "DELAY", f"{connection.delay} ticks", "conn:delay")
        self._number_row(builder, x0, 362, "GAIN", f"{connection.gain:.2f}", "conn:gain")
        self._text(builder, "NOTES", x0 + 14, 414, 0.31, (180, 185, 195))
        self._wrapped_text(builder, connection.notes, x0 + 64, 414, width_chars=42, max_lines=2, line_height=17, scale=0.27)
        self._button(builder, "edit_notes", (x0 + 14, 449, x0 + 394, 474), "EDIT ITEM NOTES [n]", scale=0.31)

    def _render_sensor_inspector(self, builder, x0, sensor_id: str) -> None:
        label = SENSORY_LABELS[sensor_id]
        self._text(builder, f"{label.upper()} SOURCE", x0 + 14, 72, 0.48, (245, 245, 245))
        if sensor_id == "sensor:brightness":
            lines = [
                "Direct luminance source for the first neural experiment.",
                "Spatial identity is measured, not reconstructed.",
                "Connect it to one or more fields to create branches.",
            ]
        else:
            lines = [
                "Fixed diagnostic map from vision_features.py.",
                "Available as comparison or optional experiment input.",
                "It is not assumed to be a biological sequential layer.",
            ]
        for index, line in enumerate(lines):
            self._text(builder, line, x0 + 14, 110 + index * 24, 0.31, (170, 175, 185))
        self._button(
            builder,
            "sensor_connect",
            (x0 + 14, 205, x0 + 394, 239),
            "START CONNECTION FROM THIS SOURCE",
            active=self.connect_mode and self.pending_source == sensor_id,
            scale=0.33,
        )

    # ------------------------------------------------------------------
    # Observation: sensory truth + future visual projection
    # ------------------------------------------------------------------

    def _camera_image(self) -> np.ndarray:
        if self.last_frame is None or not self.camera_on:
            image = self._blank(120, 200, 5)
            self._text(image, "CAMERA OFF", 58, 68, 0.52, (100, 100, 220), 2)
            return image
        return cv2.flip(self.last_frame, 1)

    def _tile(self, image, title, subtitle, width, height, *, smooth=False) -> np.ndarray:
        tile = self._blank(height, width, 7)
        title_h = 32
        fitted = self._fit_image(
            image,
            width,
            height - title_h,
            cv2.INTER_AREA if smooth else cv2.INTER_NEAREST,
        )
        tile[title_h:, :] = fitted
        cv2.rectangle(tile, (0, 0), (width - 1, title_h - 1), (20, 20, 24), -1)
        self._text(tile, title, 8, 14, 0.32, (240, 240, 245))
        self._text(tile, subtitle, 8, 28, 0.24, (145, 150, 160))
        cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (58, 58, 66), 1)
        return tile

    def _selected_signal_image(self) -> Tuple[Optional[np.ndarray], str, str]:
        if self.selected_kind == "sensor" and self.selected_id in SENSORY_SOURCES:
            if self.last_features is None:
                return None, self._display_name(self.selected_id), "no sensory frame"
            key = self.selected_id.split(":", 1)[1]
            return self._to_bgr_unit_map(self.last_features.as_dict()[key]), self._display_name(self.selected_id), "measured sensory map"

        if self.selected_kind == "layer" and self.selected_id and self.last_neural is not None:
            mode = self.spec.visualization_mode
            layer_views = self.last_neural.layer_views.get(self.selected_id, {})
            value = layer_views.get(mode)
            if value is not None:
                image = self._runtime_image(value)
                if image is not None:
                    return image, self._display_name(self.selected_id), f"real engine view: {mode}"
                return None, self._display_name(self.selected_id), "engine returned non-spatial data; not reshaped"

        return None, "SELECTED SIGNAL", "select sensory input or wait for real layer output"

    def _projection_image(self) -> Tuple[Optional[np.ndarray], str]:
        if self.last_neural is None or self.last_neural.visual_projection is None:
            return None, "No traceable network projection exists yet."
        image = self._runtime_image(self.last_neural.visual_projection)
        if image is None:
            return None, "Engine projection is not a 2D/3-channel spatial map; UI refused to reshape it."
        provenance = self.last_neural.projection_provenance.strip() or "Engine supplied traceable visual projection."
        return image, provenance

    def _render_observation(self) -> np.ndarray:
        panel = self._blank(self.OBSERVATION_H, self.WIDTH, 7)
        self._panel_title(
            panel,
            "LIVE OBSERVATION",
            "measured sensory truth above; network-derived visual output below only when justified",
            0,
            0,
            self.WIDTH,
        )

        gap = 7
        tile_w = (self.WIDTH - gap * 7) // 6
        sensory_h = 116
        items = [("WEBCAM", self._camera_image(), "physical input", True)]
        if self.last_features is not None:
            f = self.last_features
            items.extend(
                [
                    ("BRIGHTNESS", self._to_bgr_unit_map(f.brightness), "direct light", False),
                    ("CONTRAST", self._to_bgr_unit_map(f.contrast), "diagnostic", False),
                    ("MOTION", self._to_bgr_unit_map(f.motion), "diagnostic", False),
                    ("HORIZONTAL", self._to_bgr_unit_map(f.horizontal), "diagnostic", False),
                    ("VERTICAL", self._to_bgr_unit_map(f.vertical), "diagnostic", False),
                ]
            )
        while len(items) < 6:
            items.append(("WAITING", self._blank(80, 100, 4), "no frame", False))
        for index, (title, image, subtitle, smooth) in enumerate(items[:6]):
            x = gap + index * (tile_w + gap)
            panel[42:42 + sensory_h, x:x + tile_w] = self._tile(
                image, title, subtitle, tile_w, sensory_h, smooth=smooth
            )

        lower_y = 165
        half_w = (self.WIDTH - gap * 3) // 2
        lower_h = self.OBSERVATION_H - lower_y - 7

        selected_image, selected_title, selected_sub = self._selected_signal_image()
        left = self._blank(lower_h, half_w, 8)
        self._text(left, selected_title.upper(), 10, 19, 0.38, (240, 240, 245))
        self._text(left, selected_sub, 10, 36, 0.26, (150, 155, 165))
        if selected_image is not None:
            left[42:, :] = self._fit_image(selected_image, half_w, lower_h - 42)
        else:
            self._text(left, "NO SPATIAL SIGNAL TO DISPLAY", 170, 86, 0.42, (125, 135, 150))
        cv2.rectangle(left, (0, 0), (half_w - 1, lower_h - 1), (58, 58, 66), 1)

        projection, provenance = self._projection_image()
        right = self._blank(lower_h, half_w, 8)
        self._text(right, "VISUAL LOGIC OUTPUT", 10, 19, 0.38, (240, 240, 245))
        self._text(right, "must come from actual network state + spatial ancestry", 10, 36, 0.26, (150, 155, 165))
        if projection is not None:
            right[42:, : half_w // 2] = self._fit_image(projection, half_w // 2, lower_h - 42)
            self._wrapped_text(right, provenance, half_w // 2 + 12, 62, width_chars=42, max_lines=4)
        else:
            self._text(right, "NO FABRICATED RECONSTRUCTION", 150, 82, 0.42, (120, 170, 255))
            self._wrapped_text(right, provenance, 150, 105, width_chars=56, max_lines=2, scale=0.28)
        cv2.rectangle(right, (0, 0), (half_w - 1, lower_h - 1), (58, 58, 66), 1)

        panel[lower_y:lower_y + lower_h, gap:gap + half_w] = left
        x2 = gap * 2 + half_w
        panel[lower_y:lower_y + lower_h, x2:x2 + half_w] = right
        return panel

    # ------------------------------------------------------------------
    # Footer / compose
    # ------------------------------------------------------------------

    def _render_footer(self) -> np.ndarray:
        footer = self._blank(self.FOOTER_H, self.WIDTH, 11)
        cv2.line(footer, (0, 0), (self.WIDTH, 0), (65, 65, 74), 1)
        if self.note_edit_target is not None:
            self._text(footer, "EDIT NOTES - type, ENTER saves, ESC cancels", 14, 22, 0.40, (120, 205, 255))
            shown = self.note_buffer[-150:]
            self._text(footer, shown, 14, 48, 0.36, (235, 235, 240))
            return footer

        selected = (
            f"{self.selected_kind}: {self._display_name(self.selected_id)}"
            if self.selected_id is not None
            else str(self.selected_kind or "nothing selected")
        )
        self._text(
            footer,
            f"{selected} | view {self.spec.visualization_mode.upper()} | {self.status_message[:130]}",
            14,
            25,
            0.36,
            (205, 208, 215),
        )
        note = self.spec.notes if self.selected_kind == "experiment" else ""
        if note:
            self._text(footer, "NOTE: " + " ".join(note.split())[:170], 14, 50, 0.31, (155, 165, 175))
        if self.engine is None:
            self._text(footer, "ENGINE NOT CONNECTED", self.WIDTH - 205, 48, 0.34, (120, 170, 255))
        return footer

    def _compose(self) -> np.ndarray:
        self.controls.clear()
        builder = self._blank(self.BUILDER_H, self.WIDTH, 9)
        cv2.line(builder, (self.LEFT_W, 0), (self.LEFT_W, self.BUILDER_H), (55, 55, 64), 1)
        cv2.line(
            builder,
            (self.LEFT_W + self.CENTER_W, 0),
            (self.LEFT_W + self.CENTER_W, self.BUILDER_H),
            (55, 55, 64),
            1,
        )
        self._render_tools(builder)
        self._render_graph(builder)
        self._render_inspector(builder)
        return np.vstack([self._render_header(), builder, self._render_observation(), self._render_footer()])

    # ------------------------------------------------------------------
    # Selection / connection / layout
    # ------------------------------------------------------------------

    @staticmethod
    def _point_in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _distance_to_segment(point, segment) -> float:
        px, py = point
        (x1, y1), (x2, y2) = segment
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        qx, qy = x1 + t * dx, y1 + t * dy
        return math.hypot(px - qx, py - qy)

    def _select_node(self, node_id: str) -> None:
        self.selected_kind = "sensor" if node_id in SENSORY_SOURCES else "layer"
        self.selected_id = node_id

    def _handle_connect_click(self, node_id: str) -> bool:
        if not self.connect_mode:
            return False
        if self.pending_source is None:
            self.pending_source = node_id
            self._select_node(node_id)
            self.status_message = f"Connection source selected: {self._display_name(node_id)}."
            return True
        if node_id in SENSORY_SOURCES:
            self.status_message = "Connection targets are neural fields."
            return True
        try:
            connection = self.spec.add_connection(self.pending_source, node_id)
        except ValueError as exc:
            self.status_message = str(exc)
            return True
        self.connection_flash[connection.id] = time.monotonic()
        self.selected_kind, self.selected_id = "connection", connection.id
        self.connect_mode = False
        self.pending_source = None
        self._spec_changed(
            f"Connected {self._display_name(connection.source_id)} -> {self._display_name(connection.target_id)}."
        )
        return True

    def _update_drag(self, x: int, y: int) -> None:
        if self.dragging_layer_id is None:
            return
        layer = self.spec.layer_by_id(self.dragging_layer_id)
        if layer is None:
            return
        graph_x1 = self.LEFT_W + 10
        graph_y1 = 118
        graph_w = self.CENTER_W - 20
        graph_h = self.BUILDER_H - 132
        by = y - self.HEADER_H
        cx = x - self.drag_offset[0] + self.NODE_W // 2
        cy = by - self.drag_offset[1] + self.NODE_H // 2
        layer.x = float(np.clip((cx - graph_x1) / max(graph_w, 1), 0.0, 1.0))
        layer.y = float(np.clip((cy - graph_y1) / max(graph_h, 1), 0.0, 1.0))

    def _maybe_relax_layout(self) -> None:
        now = time.monotonic()
        if self.spec.layout_mode != "dynamic" or self.dragging_layer_id is not None:
            return
        if now - self._last_layout_time < 0.12:
            return
        self._last_layout_time = now
        apply_relaxation(self.spec, runtime_strengths=self._runtime_strengths(), step=0.06)

    def _reset_grid_layout(self) -> None:
        apply_grid(self.spec, unpin=True)
        self._spec_changed("Grid layout reset; all fields returned to automatic movement.")

    @staticmethod
    def _cycle(current: str, options: Tuple[str, ...]) -> str:
        index = options.index(current) if current in options else 0
        return options[(index + 1) % len(options)]

    def _selected_layer(self) -> Optional[LayerSpec]:
        return self.spec.layer_by_id(self.selected_id) if self.selected_kind == "layer" and self.selected_id else None

    def _selected_connection(self) -> Optional[ConnectionSpec]:
        return self.spec.connection_by_id(self.selected_id) if self.selected_kind == "connection" and self.selected_id else None

    def _delete_selected(self) -> None:
        if self.selected_kind == "layer" and self.selected_id:
            name = self._display_name(self.selected_id)
            if self.spec.remove_layer(self.selected_id):
                self.selected_kind = self.selected_id = None
                self._spec_changed(f"Deleted {name} and attached connections.")
                return
        if self.selected_kind == "connection" and self.selected_id:
            name = self._display_name(self.selected_id)
            if self.spec.remove_connection(self.selected_id):
                self.selected_kind = self.selected_id = None
                self._spec_changed(f"Deleted connection {name}.")
                return
        self.status_message = "Select a neural field or connection to delete it."

    def _add_branch(self) -> None:
        if self.selected_kind not in {"sensor", "layer"} or self.selected_id is None:
            self.status_message = "Select a sensory source or layer before adding a branch."
            return
        source_id = self.selected_id
        source_layer = self.spec.layer_by_id(source_id)
        rows = source_layer.rows if source_layer is not None else 100
        cols = source_layer.cols if source_layer is not None else 100
        layer = self.spec.add_layer(name=f"Branch {self.spec.next_layer_index}", rows=rows, cols=cols)
        connection = self.spec.add_connection(source_id, layer.id)
        if source_layer is not None and (source_layer.rows, source_layer.cols) == (rows, cols):
            connection.pattern = "one_to_one"
            connection.density = 1.0
        self.connection_flash[connection.id] = time.monotonic()
        self.selected_kind, self.selected_id = "layer", layer.id
        self._spec_changed(f"Created branch {self._display_name(source_id)} -> {layer.name}.")

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------

    def _note_value(self, target: Tuple[str, Optional[str]]) -> str:
        kind, item_id = target
        if kind == "experiment":
            return self.spec.notes
        if kind == "layer" and item_id:
            layer = self.spec.layer_by_id(item_id)
            return layer.notes if layer is not None else ""
        if kind == "connection" and item_id:
            connection = self.spec.connection_by_id(item_id)
            return connection.notes if connection is not None else ""
        return ""

    def _begin_note_edit(self) -> None:
        if self.selected_kind == "experiment":
            target = ("experiment", None)
        elif self.selected_kind == "layer" and self.selected_id:
            target = ("layer", self.selected_id)
        elif self.selected_kind == "connection" and self.selected_id:
            target = ("connection", self.selected_id)
        else:
            self.status_message = "Select Experiment Notes, a layer, or a connection to edit notes."
            return
        self.note_edit_target = target
        self.note_buffer = self._note_value(target)

    def _commit_note_edit(self) -> None:
        if self.note_edit_target is None:
            return
        kind, item_id = self.note_edit_target
        if kind == "experiment":
            self.spec.notes = self.note_buffer
        elif kind == "layer" and item_id:
            layer = self.spec.layer_by_id(item_id)
            if layer is not None:
                layer.notes = self.note_buffer
        elif kind == "connection" and item_id:
            connection = self.spec.connection_by_id(item_id)
            if connection is not None:
                connection.notes = self.note_buffer
        self.note_edit_target = None
        self.note_buffer = ""
        self._spec_changed("Notes updated and will be saved with the experiment JSON.")

    # ------------------------------------------------------------------
    # Save/load and controls
    # ------------------------------------------------------------------

    def _save_spec(self) -> None:
        try:
            self.spec.save(EXPERIMENT_FILE)
            self.status_message = f"Saved experiment to {EXPERIMENT_FILE}."
        except (OSError, ValueError) as exc:
            self.status_message = f"Save failed: {exc}"

    def _load_spec(self) -> None:
        if not EXPERIMENT_FILE.exists():
            self.status_message = f"No saved {EXPERIMENT_FILE} exists yet."
            return
        try:
            self.spec = ExperimentSpec.load(EXPERIMENT_FILE)
            self.selected_kind = "layer" if self.spec.layers else "experiment"
            self.selected_id = self.spec.layers[0].id if self.spec.layers else None
            self.connect_mode = False
            self.pending_source = None
            self._spec_changed(f"Loaded experiment from {EXPERIMENT_FILE}; saved layout mode resumed.")
        except (OSError, ValueError) as exc:
            self.status_message = f"Load failed: {exc}"

    def _handle_control(self, key: str) -> None:
        if key == "camera":
            self.toggle_camera()
            return
        if key == "pause":
            self.capture_paused = not self.capture_paused
            self.status_message = "Capture paused." if self.capture_paused else "Capture resumed."
            return
        if key == "reset_sensory":
            self.reset_sensory_state()
            self.status_message = "Measured temporal sensory state reset."
            return
        if key == "learning":
            if self.engine is None:
                self.status_message = "No neural engine is connected; plasticity cannot run yet."
            else:
                self.learning_on = not self.learning_on
            return
        if key == "save":
            self._save_spec()
            return
        if key == "load":
            self._load_spec()
            return
        if key == "add_layer":
            layer = self.spec.add_layer()
            self.selected_kind, self.selected_id = "layer", layer.id
            self._spec_changed(f"Added {layer.name} ({layer.rows} x {layer.cols}).")
            return
        if key == "add_branch":
            self._add_branch()
            return
        if key == "connect":
            self.connect_mode = not self.connect_mode
            self.pending_source = None
            self.status_message = "Connect mode: source then target." if self.connect_mode else "Connect mode cancelled."
            return
        if key == "delete":
            self._delete_selected()
            return
        if key == "layout_toggle":
            self.spec.set_layout_mode("manual" if self.spec.layout_mode == "dynamic" else "dynamic")
            self._spec_changed(f"Layout mode -> {self.spec.layout_mode}.")
            return
        if key == "reset_grid":
            self._reset_grid_layout()
            return
        if key == "experiment_notes":
            self.selected_kind, self.selected_id = "experiment", None
            self.status_message = "Experiment logic notes selected. Press n or click Edit Notes."
            return
        if key == "edit_notes":
            self._begin_note_edit()
            return
        if key == "pin_layer":
            layer = self._selected_layer()
            if layer is not None:
                layer.pinned = not layer.pinned
                self._spec_changed(f"{layer.name} -> {'pinned' if layer.pinned else 'automatic'} layout.")
            return
        if key == "sensor_connect" and self.selected_kind == "sensor":
            self.connect_mode = True
            self.pending_source = self.selected_id
            return
        if key.startswith("view:"):
            mode = key.split(":", 1)[1]
            self.spec.set_visualization(mode)
            self._spec_changed(f"Future neural visualization -> {mode}.")
            return

        layer = self._selected_layer()
        if layer is not None:
            if key.startswith("rows:"):
                self.spec.set_dimensions(layer.id, rows=layer.rows + int(key.split(":", 1)[1]))
                self._spec_changed(f"{layer.name} rows -> {layer.rows}.")
                return
            if key.startswith("cols:"):
                self.spec.set_dimensions(layer.id, cols=layer.cols + int(key.split(":", 1)[1]))
                self._spec_changed(f"{layer.name} cols -> {layer.cols}.")
                return
            if key.startswith("mech:"):
                mechanism = key.split(":", 1)[1]
                enabled = self.spec.toggle_mechanism(layer.id, mechanism)
                self._spec_changed(f"{layer.name}: {mechanism} {'ON' if enabled else 'OFF'} [CONFIG ONLY].")
                return

        connection = self._selected_connection()
        if connection is not None:
            if key == "conn:pattern":
                connection.pattern = self._cycle(connection.pattern, CONNECTION_PATTERNS)
            elif key == "conn:signal":
                connection.signal = self._cycle(connection.signal, SIGNAL_PROFILES)
            elif key == "conn:plasticity":
                connection.plasticity = self._cycle(connection.plasticity, PLASTICITY_RULES)
            elif key == "conn:modulator":
                connection.modulator = self._cycle(connection.modulator, MODULATOR_PROFILES)
            elif key == "conn:radius:-":
                connection.radius -= 1
            elif key == "conn:radius:+":
                connection.radius += 1
            elif key == "conn:density:-":
                connection.density -= 0.05
            elif key == "conn:density:+":
                connection.density += 0.05
            elif key == "conn:delay:-":
                connection.delay -= 1
            elif key == "conn:delay:+":
                connection.delay += 1
            elif key == "conn:gain:-":
                connection.gain -= 0.10
            elif key == "conn:gain:+":
                connection.gain += 0.10
            else:
                return
            connection.clamp()
            self._spec_changed("Connection configuration updated [CONFIG ONLY].")

    # ------------------------------------------------------------------
    # Input handling / runtime
    # ------------------------------------------------------------------

    def _mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if self.note_edit_target is not None:
            return

        if event == cv2.EVENT_MOUSEMOVE and self.dragging_layer_id is not None:
            self._update_drag(x, y)
            return
        if event == cv2.EVENT_LBUTTONUP:
            if self.dragging_layer_id is not None:
                layer = self.spec.layer_by_id(self.dragging_layer_id)
                name = self._display_name(self.dragging_layer_id)
                if layer is not None:
                    layer.pinned = True
                self.dragging_layer_id = None
                self._spec_changed(f"Moved and pinned {name}; reset grid or unpin to resume automatic pull.")
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        by = y - self.HEADER_H
        header_keys = {"camera", "pause", "reset_sensory", "learning", "save", "load"}
        for key, rect in self.controls.items():
            px, py = (x, y) if key in header_keys else (x, by)
            if self._point_in_rect(px, py, rect):
                self._handle_control(key)
                return
        if by < 0 or by >= self.BUILDER_H:
            return

        for node_id, rect in self.node_rects.items():
            if self._point_in_rect(x, by, rect):
                if self._handle_connect_click(node_id):
                    return
                self._select_node(node_id)
                self.status_message = f"Selected {self._display_name(node_id)}."
                if node_id not in SENSORY_SOURCES:
                    self.dragging_layer_id = node_id
                    self.drag_offset = (x - rect[0], by - rect[1])
                return

        nearest = None
        nearest_distance = 12.0
        for connection_id, segment in self.connection_segments.items():
            distance = self._distance_to_segment((x, by), segment)
            if distance < nearest_distance:
                nearest, nearest_distance = connection_id, distance
        if nearest is not None:
            self.selected_kind, self.selected_id = "connection", nearest
            self.status_message = f"Selected {self._display_name(nearest)}."
            return
        self.selected_kind = self.selected_id = None

    def _handle_note_key(self, key: int) -> bool:
        if self.note_edit_target is None:
            return False
        key &= 0xFF
        if key in (13, 10):
            self._commit_note_edit()
        elif key == 27:
            self.note_edit_target = None
            self.note_buffer = ""
            self.status_message = "Note edit cancelled."
        elif key in (8, 127):
            self.note_buffer = self.note_buffer[:-1]
        elif 32 <= key <= 126:
            self.note_buffer += chr(key)
        return True

    def _handle_key(self, key: int) -> bool:
        if self._handle_note_key(key):
            return True
        key &= 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord("k"):
            self.toggle_camera()
        elif key == ord("p"):
            self._handle_control("pause")
        elif key == ord("l"):
            self._handle_control("learning")
        elif key == ord("s"):
            self._save_spec()
        elif key == ord("o"):
            self._load_spec()
        elif key == ord("a"):
            self._handle_control("add_layer")
        elif key == ord("b"):
            self._handle_control("add_branch")
        elif key == ord("c"):
            self._handle_control("connect")
        elif key == ord("g"):
            self._handle_control("layout_toggle")
        elif key == ord("r"):
            self._reset_grid_layout()
        elif key == ord("e"):
            self._handle_control("experiment_notes")
        elif key == ord("n"):
            self._begin_note_edit()
        elif key in (8, 127):
            self._delete_selected()
        return True

    def reset_sensory_state(self) -> None:
        self.extractor.reset()
        if self.engine is not None:
            self.engine.reset_activity()
        self.last_features = None
        self.last_neural = None

    def run(self) -> None:
        total_h = self.HEADER_H + self.BUILDER_H + self.OBSERVATION_H + self.FOOTER_H
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.WIDTH, total_h)
        cv2.setMouseCallback(self.WINDOW, self._mouse)

        if not self.connect_camera():
            self.status_message = "Camera unavailable; experiment builder remains usable."

        print("Visual_Play neural experiment laboratory")
        print("  click/drag : select or move; manual drag pins a field")
        print("  a / b      : add field / branch")
        print("  c          : connect source -> target")
        print("  g / r      : dynamic layout toggle / reset grid")
        print("  e / n      : experiment notes / edit selected notes")
        print("  k / p      : camera connect-disconnect / pause capture")
        print("  s / o      : save / load experiment.local.json")
        print("  q / esc    : quit")

        try:
            running = True
            while running:
                now = time.time()
                dt = max(now - self._last_time, 1e-6)
                self._last_time = now
                self.fps = 0.90 * self.fps + 0.10 * (1.0 / dt)

                if self.camera_on and not self.capture_paused and self.cap is not None:
                    ok, frame = self.cap.read()
                    if ok:
                        self.last_frame = frame.copy()
                        self.last_features = self.extractor.extract(frame)
                        if self.engine is not None:
                            self.last_neural = self.engine.process(
                                self.last_features,
                                learn=self.learning_on,
                            )
                    else:
                        time.sleep(0.01)

                self._maybe_relax_layout()
                cv2.imshow(self.WINDOW, self._compose())
                running = self._handle_key(cv2.waitKey(1))
        finally:
            self.disconnect_camera()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    VisualExperimentUI().run()
