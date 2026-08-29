"""
UI.py

Visual_Play experiment-builder cockpit.

The UI owns:
- camera capture and trustworthy sensory diagnostics
- declarative experiment editing
- network-map visualization
- selection/inspection of layers and connections

The UI does NOT own neural propagation or plasticity math. A future neural
engine consumes ExperimentSpec and returns live neural state for visualization.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
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
from vision_features import VisionFeatureExtractor, VisionFeatures


EXPERIMENT_FILE = Path("experiment.local.json")


@dataclass
class NeuralFrameResult:
    """Future engine result: layer -> view-mode -> retinotopic map."""

    layer_views: Dict[str, Dict[str, np.ndarray]]
    diagnostics: Dict[str, float]


class NeuralEngine(Protocol):
    """Future runtime boundary; neural math belongs behind this interface."""

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
    WINDOW = "Visual Play - Neural Experiment Builder"

    WIDTH = 1500
    HEADER_H = 58
    BUILDER_H = 500
    SENSORY_H = 270
    FOOTER_H = 72

    LEFT_W = 220
    RIGHT_W = 420
    CENTER_W = WIDTH - LEFT_W - RIGHT_W

    NODE_W = 172
    NODE_H = 82
    SENSOR_W = 118
    SENSOR_H = 54

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

        self.fps = 0.0
        self._last_time = time.time()
        self.status_message = (
            "Builder ready: brightness -> Layer 1. Neural engine is not connected yet."
        )

        self._configure_engine()

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

    def _configure_engine(self) -> None:
        if self.engine is not None:
            self.engine.configure(self.spec)

    def _spec_changed(self, message: str) -> None:
        self.spec.validate()
        self._configure_engine()
        self.last_neural = None
        self.status_message = message

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

    def _button(
        self,
        canvas: np.ndarray,
        key: str,
        rect: Tuple[int, int, int, int],
        label: str,
        *,
        active: bool = False,
        enabled: bool = True,
        scale: float = 0.41,
    ) -> None:
        x1, y1, x2, y2 = rect
        self.controls[key] = rect

        if not enabled:
            fill = (28, 28, 32)
            border = (55, 55, 62)
            text = (100, 100, 108)
        elif active:
            fill = (55, 85, 62)
            border = (110, 210, 130)
            text = (245, 245, 245)
        else:
            fill = (38, 38, 44)
            border = (90, 90, 104)
            text = (228, 228, 232)

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
        self._text(canvas, title, x + 12, y + 24, 0.52, (245, 245, 245), 1)
        if subtitle:
            self._text(canvas, subtitle, x + 12, y + 36, 0.29, (145, 150, 162), 1)

    def _render_header(self) -> np.ndarray:
        header = self._blank(self.HEADER_H, self.WIDTH, 13)
        self._button(
            header,
            "camera",
            (12, 10, 178, 47),
            "CAMERA [k]",
            active=self.camera_on,
        )
        self._button(
            header,
            "pause",
            (188, 10, 342, 47),
            "PAUSE [p]" if not self.capture_paused else "RESUME [p]",
            active=self.capture_paused,
        )
        self._button(
            header,
            "learning",
            (352, 10, 520, 47),
            "PLASTICITY [l]",
            active=self.learning_on,
        )
        self._button(header, "save", (530, 10, 650, 47), "SAVE [s]")
        self._button(header, "load", (660, 10, 780, 47), "LOAD [o]")

        engine_status = "CONNECTED" if self.engine is not None else "NOT BUILT"
        status = (
            f"FPS {self.fps:5.1f} | layers {len(self.spec.layers)} | "
            f"links {len(self.spec.connections)} | engine {engine_status}"
        )
        self._text(header, status, 860, 34, 0.48, (218, 218, 130), 1)
        return header

    def _render_tools(self, builder: np.ndarray) -> None:
        x0 = 0
        self._panel_title(
            builder,
            "EXPERIMENT",
            "edit structure; math engine remains separate",
            x0,
            0,
            self.LEFT_W,
        )

        y = 48
        self._button(builder, "add_layer", (12, y, 208, y + 34), "+ ADD LAYER")
        y += 42
        self._button(
            builder,
            "connect",
            (12, y, 208, y + 34),
            "CONNECT NODES",
            active=self.connect_mode,
        )
        y += 42
        self._button(builder, "delete", (12, y, 208, y + 34), "DELETE SELECTED")
        y += 42
        self._button(builder, "reset_builder", (12, y, 208, y + 34), "RESET BUILDER")

        y += 54
        self._text(builder, "VISUALIZE FUTURE ENGINE", 12, y, 0.36, (170, 175, 185))
        y += 12

        col_w = 96
        for index, mode in enumerate(VISUALIZATION_MODES):
            row = index // 2
            col = index % 2
            x1 = 12 + col * (col_w + 8)
            y1 = y + row * 36
            self._button(
                builder,
                f"view:{mode}",
                (x1, y1, x1 + col_w, y1 + 28),
                mode.upper(),
                active=self.spec.visualization_mode == mode,
                scale=0.31,
            )

        info_y = 418
        if self.connect_mode:
            source = (
                self._display_name(self.pending_source)
                if self.pending_source is not None
                else "click a source"
            )
            self._text(builder, "CONNECT MODE", 12, info_y, 0.42, (120, 205, 255))
            self._text(builder, f"source: {source}", 12, info_y + 22, 0.34)
            self._text(builder, "then click a target layer", 12, info_y + 43, 0.32)
        else:
            self._text(builder, "TIP", 12, info_y, 0.40, (170, 175, 185))
            self._text(builder, "drag layers to organize flow", 12, info_y + 22, 0.32)
            self._text(builder, "click lines to edit synapses", 12, info_y + 43, 0.32)

    def _sensor_center(self, sensor_id: str) -> Tuple[int, int]:
        index = SENSORY_SOURCES.index(sensor_id)
        gap = 10
        total = len(SENSORY_SOURCES) * self.SENSOR_W + (len(SENSORY_SOURCES) - 1) * gap
        start_x = self.LEFT_W + (self.CENTER_W - total) // 2
        x1 = start_x + index * (self.SENSOR_W + gap)
        y1 = 58
        return (x1 + self.SENSOR_W // 2, y1 + self.SENSOR_H // 2)

    def _layer_rect(self, layer: LayerSpec) -> Tuple[int, int, int, int]:
        graph_x1 = self.LEFT_W + 10
        graph_y1 = 130
        graph_w = self.CENTER_W - 20
        graph_h = self.BUILDER_H - 145

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

    def _draw_arrow(
        self,
        canvas: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        *,
        selected: bool,
        signal: str,
    ) -> None:
        if selected:
            color = (110, 220, 255)
            thickness = 3
        elif signal == "inhibitory":
            color = (175, 120, 220)
            thickness = 2
        elif signal == "mixed":
            color = (110, 180, 200)
            thickness = 2
        else:
            color = (110, 165, 125)
            thickness = 2

        cv2.arrowedLine(
            canvas,
            p1,
            p2,
            color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.035,
        )

    def _render_graph(self, builder: np.ndarray) -> None:
        x0 = self.LEFT_W
        self._panel_title(
            builder,
            "NETWORK MAP",
            "sensory sources -> editable neural fields; loops and branches are allowed",
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
            self._text(
                builder,
                SENSORY_LABELS[sensor_id].upper(),
                rect[0] + 8,
                rect[1] + 24,
                0.35,
                (238, 238, 242),
            )
            tag = "DIRECT LIGHT" if sensor_id == "sensor:brightness" else "DIAGNOSTIC"
            self._text(builder, tag, rect[0] + 8, rect[1] + 43, 0.25, (145, 150, 160))

        for layer in self.spec.layers:
            self.node_rects[layer.id] = self._layer_rect(layer)

        for connection in self.spec.connections:
            p1 = self._node_center(connection.source_id)
            p2 = self._node_center(connection.target_id)
            if p1 is None or p2 is None:
                continue

            target_rect = self.node_rects.get(connection.target_id)
            if target_rect is not None:
                tx, ty = p2
                sx, sy = p1
                dx, dy = tx - sx, ty - sy
                length = max(math.hypot(dx, dy), 1.0)
                shrink = self.NODE_W * 0.34
                p2_draw = (
                    int(tx - dx / length * shrink),
                    int(ty - dy / length * (self.NODE_H * 0.34)),
                )
            else:
                p2_draw = p2

            selected = (
                self.selected_kind == "connection" and self.selected_id == connection.id
            )
            self._draw_arrow(
                builder,
                p1,
                p2_draw,
                selected=selected,
                signal=connection.signal,
            )
            self.connection_segments[connection.id] = (p1, p2_draw)

        for layer in self.spec.layers:
            rect = self.node_rects[layer.id]
            selected = self.selected_kind == "layer" and self.selected_id == layer.id
            fill = (38, 43, 48) if not selected else (50, 70, 60)
            border = (85, 95, 106) if not selected else (115, 220, 140)

            cv2.rectangle(builder, (rect[0], rect[1]), (rect[2], rect[3]), fill, -1)
            cv2.rectangle(
                builder,
                (rect[0], rect[1]),
                (rect[2], rect[3]),
                border,
                2 if selected else 1,
            )

            self._text(builder, layer.name, rect[0] + 10, rect[1] + 22, 0.48, (245, 245, 245))
            self._text(
                builder,
                f"{layer.rows} x {layer.cols} = {layer.unit_count:,} neurons",
                rect[0] + 10,
                rect[1] + 45,
                0.33,
                (175, 180, 188),
            )
            enabled = sum(1 for value in layer.mechanisms.values() if value)
            self._text(
                builder,
                f"{enabled} mechanisms | view {self.spec.visualization_mode}",
                rect[0] + 10,
                rect[1] + 65,
                0.29,
                (140, 160, 150),
            )

        if self.engine is None:
            self._text(
                builder,
                "STRUCTURE ONLY - no neural output is fabricated",
                x0 + 14,
                self.BUILDER_H - 14,
                0.34,
                (130, 135, 145),
            )

    def _render_inspector(self, builder: np.ndarray) -> None:
        x0 = self.LEFT_W + self.CENTER_W
        self._panel_title(
            builder,
            "INSPECTOR",
            "click a source, layer, or connection",
            x0,
            0,
            self.RIGHT_W,
        )

        if self.selected_kind == "layer" and self.selected_id is not None:
            layer = self.spec.layer_by_id(self.selected_id)
            if layer is not None:
                self._render_layer_inspector(builder, x0, layer)
                return

        if self.selected_kind == "connection" and self.selected_id is not None:
            connection = self.spec.connection_by_id(self.selected_id)
            if connection is not None:
                self._render_connection_inspector(builder, x0, connection)
                return

        if self.selected_kind == "sensor" and self.selected_id in SENSORY_SOURCES:
            self._render_sensor_inspector(builder, x0, self.selected_id)
            return

        self._text(builder, "Nothing selected.", x0 + 16, 76, 0.44)
        self._text(builder, "Click a graph item to edit it.", x0 + 16, 100, 0.34)

    def _dimension_row(
        self,
        builder: np.ndarray,
        x0: int,
        y: int,
        label: str,
        value: int,
        prefix: str,
    ) -> None:
        self._text(builder, label, x0 + 14, y + 21, 0.37, (175, 180, 190))
        self._button(builder, f"{prefix}:-10", (x0 + 92, y, x0 + 132, y + 28), "-10", scale=0.30)
        self._button(builder, f"{prefix}:-1", (x0 + 137, y, x0 + 174, y + 28), "-1", scale=0.30)
        cv2.rectangle(builder, (x0 + 180, y), (x0 + 258, y + 28), (26, 26, 31), -1)
        cv2.rectangle(builder, (x0 + 180, y), (x0 + 258, y + 28), (70, 70, 80), 1)
        self._text(builder, str(value), x0 + 197, y + 20, 0.39, (240, 240, 245))
        self._button(builder, f"{prefix}:+1", (x0 + 264, y, x0 + 301, y + 28), "+1", scale=0.30)
        self._button(builder, f"{prefix}:+10", (x0 + 306, y, x0 + 350, y + 28), "+10", scale=0.30)

    def _render_layer_inspector(
        self,
        builder: np.ndarray,
        x0: int,
        layer: LayerSpec,
    ) -> None:
        self._text(builder, layer.name.upper(), x0 + 14, 68, 0.50, (245, 245, 245))
        self._text(
            builder,
            f"{layer.unit_count:,} neurons | position ({layer.x:.2f}, {layer.y:.2f})",
            x0 + 14,
            91,
            0.32,
            (155, 160, 170),
        )

        self._dimension_row(builder, x0, 106, "ROWS", layer.rows, "rows")
        self._dimension_row(builder, x0, 140, "COLS", layer.cols, "cols")

        self._text(builder, "NEURON / PLASTICITY MECHANISMS", x0 + 14, 194, 0.35, (180, 185, 195))

        start_y = 207
        col_w = 188
        for index, mechanism in enumerate(MECHANISMS):
            row = index // 2
            col = index % 2
            x1 = x0 + 14 + col * (col_w + 8)
            y1 = start_y + row * 35
            self._button(
                builder,
                f"mech:{mechanism}",
                (x1, y1, x1 + col_w, y1 + 28),
                mechanism.replace("_", " ").upper(),
                active=layer.mechanisms.get(mechanism, False),
                scale=0.29,
            )

        self._text(
            builder,
            "These switches only describe intended math until the engine implements them.",
            x0 + 14,
            438,
            0.28,
            (135, 140, 150),
        )
        self._text(
            builder,
            "No checkbox is treated as cognition by the UI itself.",
            x0 + 14,
            457,
            0.28,
            (135, 140, 150),
        )

    def _cycle_row(
        self,
        builder: np.ndarray,
        x0: int,
        y: int,
        label: str,
        value: str,
        key: str,
    ) -> None:
        self._text(builder, label, x0 + 14, y + 21, 0.36, (175, 180, 190))
        self._button(
            builder,
            key,
            (x0 + 126, y, x0 + 393, y + 28),
            value.replace("_", " ").upper(),
            scale=0.32,
        )

    def _number_row(
        self,
        builder: np.ndarray,
        x0: int,
        y: int,
        label: str,
        value_text: str,
        prefix: str,
    ) -> None:
        self._text(builder, label, x0 + 14, y + 21, 0.36, (175, 180, 190))
        self._button(builder, f"{prefix}:-", (x0 + 126, y, x0 + 164, y + 28), "-", scale=0.36)
        cv2.rectangle(builder, (x0 + 170, y), (x0 + 340, y + 28), (26, 26, 31), -1)
        cv2.rectangle(builder, (x0 + 170, y), (x0 + 340, y + 28), (70, 70, 80), 1)
        self._text(builder, value_text, x0 + 184, y + 20, 0.38, (240, 240, 245))
        self._button(builder, f"{prefix}:+", (x0 + 346, y, x0 + 393, y + 28), "+", scale=0.36)

    def _render_connection_inspector(
        self,
        builder: np.ndarray,
        x0: int,
        connection: ConnectionSpec,
    ) -> None:
        self._text(builder, "CONNECTION", x0 + 14, 68, 0.50, (245, 245, 245))
        self._text(
            builder,
            f"{self._display_name(connection.source_id)}  ->  {self._display_name(connection.target_id)}",
            x0 + 14,
            92,
            0.34,
            (175, 180, 190),
        )

        self._cycle_row(builder, x0, 112, "PATTERN", connection.pattern, "conn:pattern")
        self._cycle_row(builder, x0, 148, "SIGNAL", connection.signal, "conn:signal")
        self._cycle_row(builder, x0, 184, "PLASTICITY", connection.plasticity, "conn:plasticity")
        self._cycle_row(builder, x0, 220, "MODULATOR", connection.modulator, "conn:modulator")

        self._number_row(builder, x0, 266, "RADIUS", str(connection.radius), "conn:radius")
        self._number_row(
            builder,
            x0,
            302,
            "DENSITY",
            f"{connection.density * 100:.0f}%",
            "conn:density",
        )
        self._number_row(builder, x0, 338, "DELAY", f"{connection.delay} ticks", "conn:delay")
        self._number_row(builder, x0, 374, "GAIN", f"{connection.gain:.2f}", "conn:gain")

        self._text(
            builder,
            "Profiles are explicit configuration, not claims of full neurochemistry.",
            x0 + 14,
            438,
            0.29,
            (135, 140, 150),
        )

    def _render_sensor_inspector(self, builder: np.ndarray, x0: int, sensor_id: str) -> None:
        label = SENSORY_LABELS[sensor_id]
        self._text(builder, f"{label.upper()} SOURCE", x0 + 14, 72, 0.50, (245, 245, 245))

        if sensor_id == "sensor:brightness":
            lines = [
                "Direct sensory source for the first biological experiment.",
                "Represents spatial luminance/light intensity.",
                "Use CONNECT NODES to route it into one or more neural fields.",
            ]
        else:
            lines = [
                "Precomputed diagnostic feature from vision_features.py.",
                "Useful as a comparison or optional synthetic experiment input.",
                "It is not assumed to be the biological first layer.",
            ]

        for index, line in enumerate(lines):
            self._text(builder, line, x0 + 14, 110 + index * 24, 0.32, (170, 175, 185))

        self._button(
            builder,
            "sensor_connect",
            (x0 + 14, 208, x0 + 393, 244),
            "START CONNECTION FROM THIS SOURCE",
            active=self.connect_mode and self.pending_source == sensor_id,
            scale=0.34,
        )

    def _camera_image(self) -> np.ndarray:
        if self.last_frame is None or not self.camera_on:
            image = self._blank(120, 200, 5)
            self._text(image, "CAMERA OFF", 58, 68, 0.52, (100, 100, 220), 2)
            return image
        return cv2.flip(self.last_frame, 1)

    def _sensory_tile(
        self,
        image: np.ndarray,
        title: str,
        subtitle: str,
        width: int,
        height: int,
        *,
        smooth: bool = False,
    ) -> np.ndarray:
        tile = self._blank(height, width, 7)
        title_h = 36
        fitted = self._fit_image(
            image,
            width,
            height - title_h,
            cv2.INTER_AREA if smooth else cv2.INTER_NEAREST,
        )
        tile[title_h:, :] = fitted
        cv2.rectangle(tile, (0, 0), (width - 1, title_h - 1), (20, 20, 24), -1)
        self._text(tile, title, 8, 16, 0.35, (240, 240, 245))
        self._text(tile, subtitle, 8, 31, 0.26, (145, 150, 160))
        cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (58, 58, 66), 1)
        return tile

    def _render_sensory(self) -> np.ndarray:
        panel = self._blank(self.SENSORY_H, self.WIDTH, 7)
        self._panel_title(
            panel,
            "TRUSTWORTHY SENSORY BASELINE",
            "webcam plus fixed diagnostic maps; brightness is marked as the direct neural-source baseline",
            0,
            0,
            self.WIDTH,
        )

        gap = 7
        tile_w = (self.WIDTH - gap * 7) // 6
        tile_h = self.SENSORY_H - 50

        items = [("WEBCAM", self._camera_image(), "physical input", True)]

        if self.last_features is not None:
            f = self.last_features
            items.extend(
                [
                    ("BRIGHTNESS", self._to_bgr_unit_map(f.brightness), "direct light source", False),
                    ("CONTRAST", self._to_bgr_unit_map(f.contrast), "diagnostic", False),
                    ("MOTION", self._to_bgr_unit_map(f.motion), "diagnostic", False),
                    ("HORIZONTAL", self._to_bgr_unit_map(f.horizontal), "diagnostic", False),
                    ("VERTICAL", self._to_bgr_unit_map(f.vertical), "diagnostic", False),
                ]
            )

        while len(items) < 6:
            items.append(("WAITING", self._blank(100, 100, 4), "no frame", False))

        for index, (title, image, subtitle, smooth) in enumerate(items[:6]):
            x = gap + index * (tile_w + gap)
            tile = self._sensory_tile(image, title, subtitle, tile_w, tile_h, smooth=smooth)
            panel[44:44 + tile_h, x:x + tile_w] = tile

        return panel

    def _render_footer(self) -> np.ndarray:
        footer = self._blank(self.FOOTER_H, self.WIDTH, 11)
        cv2.line(footer, (0, 0), (self.WIDTH, 0), (65, 65, 74), 1)

        selected = (
            f"{self.selected_kind}: {self._display_name(self.selected_id)}"
            if self.selected_id is not None
            else "nothing selected"
        )
        self._text(
            footer,
            f"{selected} | future neural view: {self.spec.visualization_mode.upper()}",
            14,
            25,
            0.40,
            (185, 190, 200),
        )
        self._text(footer, self.status_message[:180], 14, 52, 0.38, (215, 215, 220))

        if self.engine is None:
            self._text(
                footer,
                "ENGINE NOT CONNECTED",
                self.WIDTH - 202,
                42,
                0.38,
                (120, 170, 255),
                1,
            )
        return footer

    def _compose(self) -> np.ndarray:
        self.controls.clear()

        builder = self._blank(self.BUILDER_H, self.WIDTH, 9)
        cv2.line(
            builder,
            (self.LEFT_W, 0),
            (self.LEFT_W, self.BUILDER_H),
            (55, 55, 64),
            1,
        )
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

        return np.vstack(
            [
                self._render_header(),
                builder,
                self._render_sensory(),
                self._render_footer(),
            ]
        )

    @staticmethod
    def _point_in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _distance_to_segment(
        point: Tuple[int, int],
        segment: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> float:
        px, py = point
        (x1, y1), (x2, y2) = segment
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        qx = x1 + t * dx
        qy = y1 + t * dy
        return math.hypot(px - qx, py - qy)

    def _select_node(self, node_id: str) -> None:
        if node_id in SENSORY_SOURCES:
            self.selected_kind = "sensor"
        else:
            self.selected_kind = "layer"
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
            self.status_message = "Connection targets must currently be neural layers."
            return True

        try:
            connection = self.spec.add_connection(self.pending_source, node_id)
        except ValueError as exc:
            self.status_message = str(exc)
            return True

        self.selected_kind = "connection"
        self.selected_id = connection.id
        self.connect_mode = False
        self.pending_source = None
        self._spec_changed(
            f"Connected {self._display_name(connection.source_id)} -> "
            f"{self._display_name(connection.target_id)}."
        )
        return True

    def _update_drag(self, x: int, y: int) -> None:
        if self.dragging_layer_id is None:
            return
        layer = self.spec.layer_by_id(self.dragging_layer_id)
        if layer is None:
            return

        graph_x1 = self.LEFT_W + 10
        graph_y1 = 130
        graph_w = self.CENTER_W - 20
        graph_h = self.BUILDER_H - 145

        by = y - self.HEADER_H
        cx = x - self.drag_offset[0] + self.NODE_W // 2
        cy = by - self.drag_offset[1] + self.NODE_H // 2

        layer.x = float(np.clip((cx - graph_x1) / max(graph_w, 1), 0.0, 1.0))
        layer.y = float(np.clip((cy - graph_y1) / max(graph_h, 1), 0.0, 1.0))

    @staticmethod
    def _cycle(current: str, options: Tuple[str, ...]) -> str:
        index = options.index(current) if current in options else 0
        return options[(index + 1) % len(options)]

    def _selected_layer(self) -> Optional[LayerSpec]:
        if self.selected_kind != "layer" or self.selected_id is None:
            return None
        return self.spec.layer_by_id(self.selected_id)

    def _selected_connection(self) -> Optional[ConnectionSpec]:
        if self.selected_kind != "connection" or self.selected_id is None:
            return None
        return self.spec.connection_by_id(self.selected_id)

    def _delete_selected(self) -> None:
        if self.selected_kind == "layer" and self.selected_id is not None:
            name = self._display_name(self.selected_id)
            if self.spec.remove_layer(self.selected_id):
                self.selected_kind = None
                self.selected_id = None
                self._spec_changed(f"Deleted {name} and its attached connections.")
                return

        if self.selected_kind == "connection" and self.selected_id is not None:
            name = self._display_name(self.selected_id)
            if self.spec.remove_connection(self.selected_id):
                self.selected_kind = None
                self.selected_id = None
                self._spec_changed(f"Deleted connection {name}.")
                return

        self.status_message = "Select a neural layer or connection to delete it."

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
            self.selected_kind = "layer" if self.spec.layers else None
            self.selected_id = self.spec.layers[0].id if self.spec.layers else None
            self.connect_mode = False
            self.pending_source = None
            self._spec_changed(f"Loaded experiment from {EXPERIMENT_FILE}.")
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
        if key == "learning":
            self.learning_on = not self.learning_on
            self.status_message = (
                "Future neural plasticity enabled."
                if self.learning_on
                else "Future neural plasticity frozen."
            )
            return
        if key == "save":
            self._save_spec()
            return
        if key == "load":
            self._load_spec()
            return
        if key == "add_layer":
            layer = self.spec.add_layer()
            self.selected_kind = "layer"
            self.selected_id = layer.id
            self._spec_changed(f"Added {layer.name} ({layer.rows} x {layer.cols}).")
            return
        if key == "connect":
            self.connect_mode = not self.connect_mode
            self.pending_source = None
            self.status_message = (
                "Connect mode: click a sensory source or layer, then a target layer."
                if self.connect_mode
                else "Connect mode cancelled."
            )
            return
        if key == "delete":
            self._delete_selected()
            return
        if key == "reset_builder":
            self.spec = ExperimentSpec.default()
            self.selected_kind = "layer"
            self.selected_id = self.spec.layers[0].id
            self.connect_mode = False
            self.pending_source = None
            self._spec_changed("Builder reset to brightness -> Layer 1.")
            return
        if key == "sensor_connect" and self.selected_kind == "sensor":
            self.connect_mode = True
            self.pending_source = self.selected_id
            self.status_message = (
                f"Source fixed to {self._display_name(self.selected_id)}; click a target layer."
            )
            return

        if key.startswith("view:"):
            mode = key.split(":", 1)[1]
            self.spec.set_visualization(mode)
            self._spec_changed(f"Future neural visualization set to {mode}.")
            return

        layer = self._selected_layer()
        if layer is not None:
            if key.startswith("rows:"):
                delta = int(key.split(":", 1)[1])
                self.spec.set_dimensions(layer.id, rows=layer.rows + delta)
                self._spec_changed(f"{layer.name} rows -> {layer.rows}.")
                return
            if key.startswith("cols:"):
                delta = int(key.split(":", 1)[1])
                self.spec.set_dimensions(layer.id, cols=layer.cols + delta)
                self._spec_changed(f"{layer.name} cols -> {layer.cols}.")
                return
            if key.startswith("mech:"):
                mechanism = key.split(":", 1)[1]
                enabled = self.spec.toggle_mechanism(layer.id, mechanism)
                state = "ON" if enabled else "OFF"
                self._spec_changed(f"{layer.name}: {mechanism} {state}.")
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
            self._spec_changed(f"Updated connection: {key.split(':', 1)[1]}.")
            return

    def _mouse(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: Any,
    ) -> None:
        del flags, param

        if event == cv2.EVENT_MOUSEMOVE and self.dragging_layer_id is not None:
            self._update_drag(x, y)
            return

        if event == cv2.EVENT_LBUTTONUP:
            if self.dragging_layer_id is not None:
                name = self._display_name(self.dragging_layer_id)
                self.dragging_layer_id = None
                self._spec_changed(f"Moved {name}.")
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        by = y - self.HEADER_H
        header_keys = {"camera", "pause", "learning", "save", "load"}
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
                nearest = connection_id
                nearest_distance = distance

        if nearest is not None:
            self.selected_kind = "connection"
            self.selected_id = nearest
            self.status_message = f"Selected {self._display_name(nearest)}."
            return

        self.selected_kind = None
        self.selected_id = None

    def _handle_key(self, key: int) -> bool:
        key &= 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord("k"):
            self.toggle_camera()
        elif key == ord("p"):
            self.capture_paused = not self.capture_paused
        elif key == ord("l"):
            self.learning_on = not self.learning_on
        elif key == ord("s"):
            self._save_spec()
        elif key == ord("o"):
            self._load_spec()
        elif key == ord("a"):
            self._handle_control("add_layer")
        elif key == ord("c"):
            self._handle_control("connect")
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
        total_h = self.HEADER_H + self.BUILDER_H + self.SENSORY_H + self.FOOTER_H
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.WIDTH, total_h)
        cv2.setMouseCallback(self.WINDOW, self._mouse)

        if not self.connect_camera():
            self.status_message = "Camera unavailable; builder still works with camera OFF."

        print("Visual_Play neural experiment builder")
        print("  click      : select / edit")
        print("  drag       : move neural layers on the network map")
        print("  a          : add layer")
        print("  c          : connect source -> target")
        print("  k          : connect / disconnect camera")
        print("  p          : pause / resume capture")
        print("  l          : toggle future plasticity")
        print("  s / o      : save / load experiment.local.json")
        print("  q / esc    : quit")

        try:
            running = True
            while running:
                now = time.time()
                dt = max(now - self._last_time, 1e-6)
                self._last_time = now
                self.fps = 0.90 * self.fps + 0.10 * (1.0 / dt)

                if (
                    self.camera_on
                    and not self.capture_paused
                    and self.cap is not None
                ):
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

                cv2.imshow(self.WINDOW, self._compose())
                running = self._handle_key(cv2.waitKey(1))
        finally:
            self.disconnect_camera()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    VisualExperimentUI().run()
