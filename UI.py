"""3D node-centric Visual_Play laboratory.

The central canvas renders neuron fields as real 2D sheets in a 3D structural
space:

    X/Y = position inside a neuron field
    Z   = feed-forward depth

Ordinary downstream fields advance along Z. Explicit branches may share a depth
and split laterally for visibility. Whole fields never move because of
plasticity; future learned motion belongs to individual neuron positions and
synapses returned by the neural engine.
"""

from __future__ import annotations

import math
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Protocol

import cv2
import numpy as np

from experiment_spec import (
    MECHANISMS,
    SENSORY_LABELS,
    SENSORY_SOURCES,
    VISUALIZATION_MODES,
    ExperimentSpec,
)
from graph_layout import (
    field_depths,
    field_world_positions,
    plane_corners,
    plane_node_positions,
    project_points_3d,
)
from vision_features import VisionFeatureExtractor, VisionFeatures


EXPERIMENT_FILE = Path("experiment.local.json")


@dataclass
class NeuralFrameResult:
    layer_views: Dict[str, Dict[str, np.ndarray]]
    diagnostics: Dict[str, float]
    visual_projection: Optional[np.ndarray] = None
    projection_provenance: str = ""
    # Optional future runtime ownership:
    # Nx2 = local normalized XY positions on the field plane.
    # Nx3 = absolute world XYZ positions for individual neurons.
    node_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    synapses: Dict[str, np.ndarray] = field(default_factory=dict)


class NeuralEngine(Protocol):
    def configure(self, spec: ExperimentSpec) -> None: ...

    def process(
        self,
        features: VisionFeatures,
        *,
        learn: bool = True,
    ) -> NeuralFrameResult: ...

    def reset_activity(self) -> None: ...


class VisualExperimentUI:
    WINDOW = "Visual Play - 3D Node Laboratory"

    WIDTH = 1500
    HEADER_H = 58
    BUILDER_H = 500
    OBS_H = 300
    FOOTER_H = 62

    LEFT_W = 220
    RIGHT_W = 360
    CENTER_W = WIDTH - LEFT_W - RIGHT_W

    FIELD_WIDTH = 3.6
    FIELD_HEIGHT = 2.4

    def __init__(self, extractor=None, engine=None, spec=None):
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

        self.cap = None
        self.camera_on = False
        self.paused = False
        self.learning_on = True

        self.last_frame = None
        self.last_features = None
        self.last_neural = None

        self.selected_kind = "layer"
        self.selected_id = self.spec.layers[0].id if self.spec.layers else None

        self.connect_mode = False
        self.pending_source = None

        self.controls = {}
        self.sensor_rects = {}
        self.field_rects = {}
        self.field_hulls = {}
        self.field_screen_nodes = {}
        self.path_segments = {}

        self.note_target = None
        self.note_buffer = ""

        self.view_yaw = 32.0
        self.view_pitch = -16.0
        self.view_dragging = False
        self.view_drag_last = (0, 0)

        self.fps = 0.0
        self._last = time.time()
        self.status = "3D node view ready: downstream fields advance on Z."

        if engine:
            engine.configure(self.spec)

    # ------------------------------------------------------------------
    # Small rendering helpers
    # ------------------------------------------------------------------

    def _blank(self, h, w, v=9):
        return np.full((h, w, 3), v, np.uint8)

    def _text(self, canvas, text, x, y, scale=0.38, color=(225, 225, 230), thickness=1):
        cv2.putText(
            canvas,
            str(text),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _button(self, canvas, key, rect, label, active=False, enabled=True, scale=0.32):
        self.controls[key] = rect
        x1, y1, x2, y2 = rect
        fill = (48, 78, 54) if active else (38, 34, 35)
        border = (105, 210, 125) if active else (82, 76, 78)
        text = (235, 235, 240)
        if not enabled:
            fill, border, text = (25, 25, 29), (50, 50, 56), (95, 95, 102)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)
        self._text(canvas, label, x1 + 7, y1 + (y2 - y1) // 2 + 4, scale, text)

    def _panel(self, canvas, title, subtitle, x, y, width):
        cv2.rectangle(canvas, (x, y), (x + width - 1, y + 38), (19, 17, 18), -1)
        self._text(canvas, title, x + 10, y + 22, 0.46)
        self._text(canvas, subtitle, x + 10, y + 35, 0.25, (140, 145, 154))

    def _wrap(self, canvas, text, x, y, width=44, lines=4, scale=0.27):
        cleaned = " ".join(str(text).split())
        for index, line in enumerate(textwrap.wrap(cleaned, width=width)[:lines]):
            self._text(canvas, line, x, y + index * 17, scale, (160, 165, 175))

    # ------------------------------------------------------------------
    # Camera ownership
    # ------------------------------------------------------------------

    def _open_camera(self):
        modes = []
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            modes.append((0, cv2.CAP_AVFOUNDATION))
        modes.append((0, cv2.CAP_ANY))

        for index, backend in modes:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            cap.release()
        return None

    def toggle_camera(self):
        if self.camera_on:
            if self.cap:
                self.cap.release()
            self.cap = None
            self.camera_on = False
            self.status = "Camera disconnected."
            return

        self.cap = self._open_camera()
        self.camera_on = bool(self.cap and self.cap.isOpened())
        self.status = "Camera connected." if self.camera_on else "Camera unavailable."

    def _changed(self, message):
        self.spec.validate()
        self.last_neural = None
        self.status = message
        if self.engine:
            self.engine.configure(self.spec)

    # ------------------------------------------------------------------
    # Header and controls
    # ------------------------------------------------------------------

    def _header(self):
        canvas = self._blank(self.HEADER_H, self.WIDTH, 12)
        self._button(canvas, "camera", (12, 10, 170, 47), "CAMERA [k]", self.camera_on)
        self._button(
            canvas,
            "pause",
            (180, 10, 330, 47),
            "RESUME [p]" if self.paused else "PAUSE [p]",
            self.paused,
        )
        self._button(canvas, "reset", (340, 10, 480, 47), "RESET INPUT")
        self._button(
            canvas,
            "learning",
            (490, 10, 640, 47),
            "PLASTICITY [l]",
            self.learning_on and self.engine is not None,
            self.engine is not None,
        )
        self._button(canvas, "save", (650, 10, 755, 47), "SAVE [s]")
        self._button(canvas, "load", (765, 10, 870, 47), "LOAD [o]")

        self._text(
            canvas,
            (
                f"FPS {self.fps:4.1f} | fields {len(self.spec.layers)} | "
                f"paths {len(self.spec.connections)} | "
                f"3D yaw {self.view_yaw:.0f} pitch {self.view_pitch:.0f} | "
                f"engine {'CONNECTED' if self.engine else 'NOT BUILT'}"
            ),
            900,
            34,
            0.35,
            (160, 220, 220),
        )
        return canvas

    def _tools(self, canvas):
        self._panel(canvas, "EXPERIMENT", "3D neuron fields and pathways", 0, 0, self.LEFT_W)
        y = 48

        actions = [
            ("add", "+ NEXT FIELD [a]"),
            ("branch", "+ BRANCH [b]"),
            ("connect", "CONNECT PATH [c]"),
            ("delete", "DELETE SELECTED"),
            ("view_reset", "RESET 3D VIEW [v]"),
            ("notes", "EXPERIMENT NOTES [e]"),
        ]
        for key, label in actions:
            self._button(
                canvas,
                key,
                (12, y, self.LEFT_W - 12, y + 32),
                label,
                active=(key == "connect" and self.connect_mode)
                or (key == "notes" and self.selected_kind == "experiment"),
            )
            y += 38

        self._text(canvas, "ENGINE VIEW", 12, y + 8, 0.30, (165, 170, 180))
        y += 17

        for index, mode in enumerate(VISUALIZATION_MODES):
            row, col = divmod(index, 2)
            x = 12 + col * 98
            yy = y + row * 29
            self._button(
                canvas,
                f"view:{mode}",
                (x, yy, x + 92, yy + 23),
                mode.upper(),
                self.spec.visualization_mode == mode,
                scale=0.23,
            )

        self._wrap(
            canvas,
            "Right-drag the 3D canvas to rotate. X/Y live inside each field; Z is network depth.",
            12,
            self.BUILDER_H - 58,
            width=29,
            lines=3,
            scale=0.24,
        )

    # ------------------------------------------------------------------
    # 3D structural view
    # ------------------------------------------------------------------

    def _screen_center(self):
        return (
            self.LEFT_W + self.CENTER_W // 2,
            292,
        )

    def _project(self, points):
        return project_points_3d(
            points,
            screen_center=self._screen_center(),
            yaw_degrees=self.view_yaw,
            pitch_degrees=self.view_pitch,
            camera_distance=8.0,
            focal_length=520.0,
        )

    def _world_nodes(self, layer, center):
        """Return one real/display XYZ position per neuron."""
        if self.last_neural is not None:
            raw = np.asarray(
                self.last_neural.node_positions.get(layer.id, []),
                dtype=np.float32,
            )

            # Absolute runtime XYZ positions.
            if raw.ndim == 2 and raw.shape == (layer.unit_count, 3):
                if np.all(np.isfinite(raw)):
                    return raw

            # Local normalized XY runtime positions on this structural Z plane.
            if raw.ndim == 2 and raw.shape == (layer.unit_count, 2):
                if np.all(np.isfinite(raw)):
                    local = np.clip(raw, 0.0, 1.0)
                    cx, cy, cz = center
                    return np.column_stack(
                        [
                            cx + (local[:, 0] - 0.5) * self.FIELD_WIDTH,
                            cy + (local[:, 1] - 0.5) * self.FIELD_HEIGHT,
                            np.full(layer.unit_count, cz, dtype=np.float32),
                        ]
                    ).astype(np.float32)

        return plane_node_positions(
            layer.rows,
            layer.cols,
            center,
            width=self.FIELD_WIDTH,
            height=self.FIELD_HEIGHT,
        )

    def _sensor_nodes(self, canvas):
        start = self.LEFT_W + 14
        gap = 7
        width = (self.CENTER_W - 28 - gap * 4) // 5
        nodes = {}
        self.sensor_rects = {}

        for index, sensor_id in enumerate(SENSORY_SOURCES):
            x1 = start + index * (width + gap)
            rect = (x1, 44, x1 + width, 76)
            self.sensor_rects[sensor_id] = rect
            selected = self.selected_kind == "sensor" and self.selected_id == sensor_id

            cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (38, 42, 44), -1)
            cv2.rectangle(
                canvas,
                (rect[0], rect[1]),
                (rect[2], rect[3]),
                (110, 205, 235) if selected else (75, 85, 92),
                2 if selected else 1,
            )
            self._text(canvas, SENSORY_LABELS[sensor_id].upper(), rect[0] + 5, 64, 0.26)

            xs = np.linspace(
                (rect[0] + rect[2]) // 2 - 28,
                (rect[0] + rect[2]) // 2 + 28,
                16,
                dtype=np.int32,
            )
            points = np.column_stack([xs, np.full_like(xs, 82)])
            nodes[sensor_id] = points
            canvas[points[:, 1], points[:, 0]] = (145, 175, 185)

        return nodes

    @staticmethod
    def _rect_from_points(points):
        if len(points) == 0:
            return (0, 0, 0, 0)
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
        return int(x1), int(y1), int(x2), int(y2)

    def _draw_field(self, canvas, layer, center, depth):
        world_nodes = self._world_nodes(layer, center)
        screen_nodes, camera_depth = self._project(world_nodes)

        corners_world = plane_corners(
            center,
            width=self.FIELD_WIDTH,
            height=self.FIELD_HEIGHT,
        )
        hull, hull_depth = self._project(corners_world)
        hull = hull.reshape(-1, 1, 2).astype(np.int32)

        self.field_hulls[layer.id] = hull
        self.field_rects[layer.id] = self._rect_from_points(hull.reshape(-1, 2))
        self.field_screen_nodes[layer.id] = screen_nodes

        selected = self.selected_kind == "layer" and self.selected_id == layer.id
        border = (110, 220, 135) if selected else (55, 70, 65)

        # The outline is only the structural sheet boundary, not a layer card.
        cv2.polylines(canvas, [hull], True, border, 2 if selected else 1, cv2.LINE_AA)

        valid = (
            (screen_nodes[:, 0] >= self.LEFT_W)
            & (screen_nodes[:, 0] < self.LEFT_W + self.CENTER_W)
            & (screen_nodes[:, 1] >= 40)
            & (screen_nodes[:, 1] < self.BUILDER_H - 12)
        )
        visible = screen_nodes[valid]
        if len(visible):
            canvas[visible[:, 1], visible[:, 0]] = (
                (175, 225, 185) if selected else (135, 175, 148)
            )

        # Close fields may support tiny circles; 10k-node fields stay pixel-cheap.
        if layer.unit_count <= 1600:
            for x, y in visible:
                cv2.circle(canvas, (int(x), int(y)), 1, (165, 205, 174), -1)

        label_anchor = hull.reshape(-1, 2)[0]
        self._text(
            canvas,
            f"Z{depth} · {layer.name} · {layer.rows}×{layer.cols}",
            label_anchor[0] + 4,
            label_anchor[1] - 6,
            0.25,
            (185, 195, 190),
        )

        return screen_nodes, float(np.mean(hull_depth))

    def _draw_path(self, canvas, path, source, target):
        if not len(source) or not len(target):
            return

        count = min(42, len(source), len(target))
        source_indices = np.linspace(0, len(source) - 1, count, dtype=int)

        if path.pattern == "one_to_one":
            target_indices = np.linspace(0, len(target) - 1, count, dtype=int)
        else:
            target_indices = (source_indices * 37 + 11) % len(target)

        color = (175, 120, 220) if path.signal == "inhibitory" else (90, 145, 100)
        if self.selected_kind == "connection" and self.selected_id == path.id:
            color = (105, 220, 245)

        for source_index, target_index in zip(source_indices, target_indices):
            cv2.line(
                canvas,
                tuple(map(int, source[source_index])),
                tuple(map(int, target[target_index])),
                color,
                1,
                cv2.LINE_AA,
            )

        start = tuple(np.mean(source, axis=0).astype(int))
        end = tuple(np.mean(target, axis=0).astype(int))
        self.path_segments[path.id] = (start, end)

    def _draw_axes(self, canvas):
        origin = np.asarray([[0.0, -1.8, 0.0]], dtype=np.float32)
        axes = np.asarray(
            [
                [1.0, -1.8, 0.0],
                [0.0, -0.8, 0.0],
                [0.0, -1.8, 1.3],
            ],
            dtype=np.float32,
        )
        origin_screen, _ = self._project(origin)
        axes_screen, _ = self._project(axes)
        origin_xy = tuple(origin_screen[0])

        labels = ("X", "Y", "Z depth")
        for point, label in zip(axes_screen, labels):
            cv2.arrowedLine(
                canvas,
                origin_xy,
                tuple(point),
                (90, 105, 115),
                1,
                cv2.LINE_AA,
                tipLength=0.10,
            )
            self._text(canvas, label, point[0] + 3, point[1] - 3, 0.22, (135, 145, 155))

    def _graph(self, canvas):
        self._panel(
            canvas,
            "3D NEURON / SYNAPSE SPACE",
            "X/Y = neuron position inside field · Z = depth · right-drag rotates",
            self.LEFT_W,
            0,
            self.CENTER_W,
        )

        self.path_segments = {}
        self.field_hulls = {}
        self.field_rects = {}
        self.field_screen_nodes = {}

        sensor_nodes = self._sensor_nodes(canvas)
        world_centers = field_world_positions(self.spec)
        depths = field_depths(self.spec) if self.spec.layers else {}

        # Project once to establish far-to-near draw order.
        draw_order = []
        for layer in self.spec.layers:
            center = world_centers.get(layer.id, (0.0, 0.0, 0.0))
            corners = plane_corners(
                center,
                width=self.FIELD_WIDTH,
                height=self.FIELD_HEIGHT,
            )
            _, camera_depth = self._project(corners)
            draw_order.append((float(np.mean(camera_depth)), layer))

        projected = {}

        # Far fields first, near fields last.
        for _, layer in sorted(draw_order, key=lambda item: item[0], reverse=True):
            center = world_centers.get(layer.id, (0.0, 0.0, 0.0))
            projected[layer.id], _ = self._draw_field(
                canvas,
                layer,
                center,
                depths.get(layer.id, 0),
            )

        # Draw pathways over field planes. These are sampled configured routes,
        # not claims of learned synapses.
        for path in self.spec.connections:
            source = sensor_nodes.get(
                path.source_id,
                projected.get(path.source_id, np.empty((0, 2), dtype=np.int32)),
            )
            target = projected.get(
                path.target_id,
                np.empty((0, 2), dtype=np.int32),
            )
            self._draw_path(canvas, path, source, target)

        self._draw_axes(canvas)
        self._text(
            canvas,
            "FIELDS ARE STRUCTURAL Z PLANES; FUTURE PLASTICITY MOVES INDIVIDUAL NODES/SYNAPSES",
            self.LEFT_W + 12,
            self.BUILDER_H - 9,
            0.24,
            (130, 135, 145),
        )

    # ------------------------------------------------------------------
    # Inspector
    # ------------------------------------------------------------------

    def _inspector(self, canvas):
        x = self.LEFT_W + self.CENTER_W
        self._panel(canvas, "INSPECTOR", "selected field/path", x, 0, self.RIGHT_W)

        if self.selected_kind == "layer" and self.selected_id:
            layer = self.spec.layer_by_id(self.selected_id)
            if layer:
                depth = field_depths(self.spec).get(layer.id, 0)
                self._text(canvas, layer.name.upper(), x + 12, 68, 0.46)
                self._text(
                    canvas,
                    f"{layer.unit_count:,} neurons · structural Z depth {depth}",
                    x + 12,
                    91,
                    0.28,
                    (155, 165, 170),
                )
                self._text(canvas, f"ROWS {layer.rows}   COLS {layer.cols}", x + 12, 120, 0.32)
                self._text(canvas, "MECHANISMS — CONFIG ONLY", x + 12, 156, 0.29, (180, 185, 195))

                start_y = 172
                for index, mechanism in enumerate(MECHANISMS):
                    row, col = divmod(index, 2)
                    xx = x + 12 + col * 168
                    yy = start_y + row * 31
                    self._button(
                        canvas,
                        f"mech:{mechanism}",
                        (xx, yy, xx + 160, yy + 25),
                        mechanism.replace("_", " ").upper(),
                        layer.mechanisms.get(mechanism, False),
                        scale=0.23,
                    )

                self._text(canvas, "NOTES", x + 12, 384, 0.28, (165, 170, 180))
                self._wrap(canvas, layer.notes, x + 12, 405, width=48, lines=2, scale=0.24)
                self._button(
                    canvas,
                    "edit_note",
                    (x + 12, 456, x + self.RIGHT_W - 12, 487),
                    "EDIT FIELD NOTES [n]",
                )
                return

        if self.selected_kind == "connection" and self.selected_id:
            path = self.spec.connection_by_id(self.selected_id)
            if path:
                self._text(canvas, "PATHWAY", x + 12, 68, 0.46)
                self._text(
                    canvas,
                    f"{self._name(path.source_id)} → {self._name(path.target_id)}",
                    x + 12,
                    93,
                    0.29,
                )
                self._text(
                    canvas,
                    f"pattern {path.pattern} | signal {path.signal}",
                    x + 12,
                    122,
                    0.29,
                )
                self._wrap(
                    canvas,
                    "Canvas lines are sampled configured routes. Exact learned synapses come only from a real engine.",
                    x + 12,
                    160,
                    width=49,
                    lines=5,
                )
                self._button(
                    canvas,
                    "edit_note",
                    (x + 12, 456, x + self.RIGHT_W - 12, 487),
                    "EDIT PATH NOTES [n]",
                )
                return

        if self.selected_kind == "experiment":
            self._text(canvas, "EXPERIMENT LOGIC", x + 12, 68, 0.46)
            self._wrap(canvas, self.spec.notes, x + 12, 105, width=50, lines=12)
            self._button(
                canvas,
                "edit_note",
                (x + 12, 456, x + self.RIGHT_W - 12, 487),
                "EDIT NOTES [n]",
            )

    def _name(self, item_id):
        if item_id in SENSORY_LABELS:
            return SENSORY_LABELS[item_id]
        layer = self.spec.layer_by_id(item_id)
        if layer:
            return layer.name
        path = self.spec.connection_by_id(item_id)
        if path:
            return f"{self._name(path.source_id)} → {self._name(path.target_id)}"
        return str(item_id)

    # ------------------------------------------------------------------
    # Observation surface
    # ------------------------------------------------------------------

    @staticmethod
    def _map(array):
        array = np.clip(np.nan_to_num(np.asarray(array, np.float32)), 0, 1)
        return cv2.cvtColor((array * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def _tile(self, image, title, subtitle, width, height, smooth=False):
        tile = self._blank(height, width, 7)
        header_h = 31
        tile[header_h:] = cv2.resize(
            image,
            (width, height - header_h),
            interpolation=cv2.INTER_AREA if smooth else cv2.INTER_NEAREST,
        )
        cv2.rectangle(tile, (0, 0), (width - 1, header_h - 1), (20, 20, 24), -1)
        self._text(tile, title, 6, 14, 0.29)
        self._text(tile, subtitle, 6, 27, 0.22, (145, 150, 160))
        return tile

    def _observation(self):
        canvas = self._blank(self.OBS_H, self.WIDTH, 7)
        self._panel(
            canvas,
            "LIVE OBSERVATION",
            "measured input and future network-derived output",
            0,
            0,
            self.WIDTH,
        )

        gap = 7
        tile_w = (self.WIDTH - gap * 7) // 6
        sensory_h = 116

        camera = (
            cv2.flip(self.last_frame, 1)
            if self.last_frame is not None and self.camera_on
            else self._blank(80, 100, 4)
        )
        items = [("WEBCAM", camera, "physical input", True)]

        if self.last_features is not None:
            features = self.last_features
            items += [
                ("BRIGHTNESS", self._map(features.brightness), "direct light", False),
                ("CONTRAST", self._map(features.contrast), "diagnostic", False),
                ("MOTION", self._map(features.motion), "diagnostic", False),
                ("HORIZONTAL", self._map(features.horizontal), "diagnostic", False),
                ("VERTICAL", self._map(features.vertical), "diagnostic", False),
            ]

        while len(items) < 6:
            items.append(("WAITING", self._blank(80, 100, 4), "no frame", False))

        for index, (title, image, subtitle, smooth) in enumerate(items[:6]):
            x = gap + index * (tile_w + gap)
            canvas[42 : 42 + sensory_h, x : x + tile_w] = self._tile(
                image,
                title,
                subtitle,
                tile_w,
                sensory_h,
                smooth,
            )

        lower_y = 165
        half_w = (self.WIDTH - gap * 3) // 2
        lower_h = self.OBS_H - lower_y - 7

        left = self._blank(lower_h, half_w, 8)
        right = self._blank(lower_h, half_w, 8)
        self._text(left, "SELECTED SIGNAL", 10, 19, 0.36)
        self._text(right, "VISUAL LOGIC OUTPUT", 10, 19, 0.36)
        self._text(left, "NO NEURAL SIGNAL YET", 185, 84, 0.38, (125, 135, 150))
        self._text(right, "NO FABRICATED RECONSTRUCTION", 145, 84, 0.38, (120, 170, 255))

        canvas[lower_y : lower_y + lower_h, gap : gap + half_w] = left
        canvas[
            lower_y : lower_y + lower_h,
            gap * 2 + half_w : gap * 2 + half_w * 2,
        ] = right
        return canvas

    # ------------------------------------------------------------------
    # Experiment editing
    # ------------------------------------------------------------------

    def _default_source(self):
        if self.selected_kind in {"sensor", "layer"} and self.selected_id:
            return self.selected_id

        if self.spec.layers:
            depths = field_depths(self.spec)
            return max(
                self.spec.layers,
                key=lambda layer: (depths.get(layer.id, 0), layer.id),
            ).id

        return "sensor:brightness"

    def _add_downstream(self, *, branch=False):
        source_id = self._default_source()
        source_layer = self.spec.layer_by_id(source_id)

        next_index = self.spec.next_layer_index
        layer = self.spec.add_layer(
            name=f"{'Branch' if branch else 'Field'} {next_index}",
            rows=source_layer.rows if source_layer else 100,
            cols=source_layer.cols if source_layer else 100,
        )
        path = self.spec.add_connection(source_id, layer.id)

        if source_layer is not None:
            path.pattern = "one_to_one"
            path.density = 1.0

        self.selected_kind = "layer"
        self.selected_id = layer.id
        self._changed(
            f"{'Branch' if branch else 'Next field'} created from {self._name(source_id)} at deeper Z."
        )

    def _begin_note(self):
        if self.selected_kind == "experiment":
            self.note_target = ("experiment", None)
            self.note_buffer = self.spec.notes
        elif self.selected_kind == "layer" and self.selected_id:
            self.note_target = ("layer", self.selected_id)
            self.note_buffer = self.spec.layer_by_id(self.selected_id).notes
        elif self.selected_kind == "connection" and self.selected_id:
            self.note_target = ("connection", self.selected_id)
            self.note_buffer = self.spec.connection_by_id(self.selected_id).notes

    def _commit_note(self):
        kind, item_id = self.note_target
        if kind == "experiment":
            self.spec.notes = self.note_buffer
        elif kind == "layer":
            self.spec.layer_by_id(item_id).notes = self.note_buffer
        else:
            self.spec.connection_by_id(item_id).notes = self.note_buffer

        self.note_target = None
        self.note_buffer = ""
        self._changed("Notes updated.")

    def _control(self, key):
        if key == "camera":
            self.toggle_camera()
        elif key == "pause":
            self.paused = not self.paused
        elif key == "reset":
            self.extractor.reset()
            self.last_features = None
            self.last_neural = None
        elif key == "learning":
            if self.engine:
                self.learning_on = not self.learning_on
        elif key == "save":
            self.spec.save(EXPERIMENT_FILE)
            self.status = "Saved experiment."
        elif key == "load":
            self.spec = ExperimentSpec.load(EXPERIMENT_FILE)
            self.selected_kind = "layer"
            self.selected_id = self.spec.layers[0].id if self.spec.layers else None
            self._changed("Loaded experiment.")
        elif key == "add":
            self._add_downstream(branch=False)
        elif key == "branch":
            self._add_downstream(branch=True)
        elif key == "connect":
            self.connect_mode = not self.connect_mode
            self.pending_source = None
        elif key == "delete":
            if self.selected_kind == "layer" and self.selected_id:
                self.spec.remove_layer(self.selected_id)
            elif self.selected_kind == "connection" and self.selected_id:
                self.spec.remove_connection(self.selected_id)
            self.selected_kind = None
            self.selected_id = None
            self._changed("Deleted selection.")
        elif key == "view_reset":
            self.view_yaw = 32.0
            self.view_pitch = -16.0
            self.status = "3D view reset."
        elif key == "notes":
            self.selected_kind = "experiment"
            self.selected_id = None
        elif key == "edit_note":
            self._begin_note()
        elif key.startswith("view:"):
            self.spec.set_visualization(key.split(":", 1)[1])
        elif key.startswith("mech:") and self.selected_kind == "layer":
            self.spec.toggle_mechanism(self.selected_id, key.split(":", 1)[1])
            self._changed("Mechanism changed [NOT SIMULATED].")

    # ------------------------------------------------------------------
    # Mouse / keyboard
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(point, segment):
        px, py = point
        (x1, y1), (x2, y2) = segment
        dx = x2 - x1
        dy = y2 - y1
        if not dx and not dy:
            return math.hypot(px - x1, py - y1)
        t = max(
            0.0,
            min(
                1.0,
                ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy),
            ),
        )
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    def _select(self, item_id):
        self.selected_kind = "sensor" if item_id in SENSORY_SOURCES else "layer"
        self.selected_id = item_id

    def _node_click(self, item_id):
        if not self.connect_mode:
            self._select(item_id)
            return

        if self.pending_source is None:
            self.pending_source = item_id
            self._select(item_id)
            return

        if item_id in SENSORY_SOURCES:
            self.status = "Target must be a field."
            return

        try:
            path = self.spec.add_connection(self.pending_source, item_id)
            self.selected_kind = "connection"
            self.selected_id = path.id
            self.connect_mode = False
            self.pending_source = None
            self._changed("Path configured.")
        except ValueError as exc:
            self.status = str(exc)

    def _mouse(self, event, x, y, flags, param):
        del flags, param
        if self.note_target:
            return

        by = y - self.HEADER_H

        # Right drag rotates the 3D camera; it never moves fields or neurons.
        if event == cv2.EVENT_RBUTTONDOWN:
            if 0 <= by < self.BUILDER_H:
                self.view_dragging = True
                self.view_drag_last = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.view_dragging:
            dx = x - self.view_drag_last[0]
            dy = y - self.view_drag_last[1]
            self.view_drag_last = (x, y)
            self.view_yaw = (self.view_yaw + dx * 0.25) % 360.0
            self.view_pitch = float(np.clip(self.view_pitch + dy * 0.20, -60.0, 60.0))
            return

        if event == cv2.EVENT_RBUTTONUP:
            self.view_dragging = False
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        header_keys = {"camera", "pause", "reset", "learning", "save", "load"}
        for key, rect in self.controls.items():
            px, py = (x, y) if key in header_keys else (x, by)
            if rect[0] <= px <= rect[2] and rect[1] <= py <= rect[3]:
                self._control(key)
                return

        for sensor_id, rect in self.sensor_rects.items():
            if rect[0] <= x <= rect[2] and rect[1] <= by <= rect[3]:
                self._node_click(sensor_id)
                return

        # Hit-test projected 3D field hulls, nearest fields first.
        world = field_world_positions(self.spec)
        ordered = sorted(
            self.spec.layers,
            key=lambda layer: world.get(layer.id, (0.0, 0.0, 0.0))[2],
        )
        for layer in ordered:
            hull = self.field_hulls.get(layer.id)
            if hull is None:
                continue
            if cv2.pointPolygonTest(hull, (float(x), float(by)), False) >= 0:
                self._node_click(layer.id)
                return

        best = (None, 10.0)
        for connection_id, segment in self.path_segments.items():
            distance = self._distance((x, by), segment)
            if distance < best[1]:
                best = (connection_id, distance)

        if best[0]:
            self.selected_kind = "connection"
            self.selected_id = best[0]

    def _key(self, key):
        key &= 255

        if self.note_target:
            if key in (13, 10):
                self._commit_note()
            elif key == 27:
                self.note_target = None
                self.note_buffer = ""
            elif key in (8, 127):
                self.note_buffer = self.note_buffer[:-1]
            elif 32 <= key <= 126:
                self.note_buffer += chr(key)
            return True

        if key in (ord("q"), 27):
            return False

        mapping = {
            ord("k"): "camera",
            ord("p"): "pause",
            ord("l"): "learning",
            ord("s"): "save",
            ord("o"): "load",
            ord("a"): "add",
            ord("b"): "branch",
            ord("c"): "connect",
            ord("v"): "view_reset",
            ord("e"): "notes",
            ord("n"): "edit_note",
        }
        if key in mapping:
            self._control(mapping[key])
        return True

    # ------------------------------------------------------------------
    # Compose and runtime
    # ------------------------------------------------------------------

    def _footer(self):
        canvas = self._blank(self.FOOTER_H, self.WIDTH, 11)
        if self.note_target:
            self._text(
                canvas,
                "EDIT NOTES — ENTER saves, ESC cancels",
                14,
                21,
                0.36,
                (120, 205, 255),
            )
            self._text(canvas, self.note_buffer[-170:], 14, 48, 0.32)
            return canvas

        selected = self._name(self.selected_id) if self.selected_id else ""
        self._text(
            canvas,
            f"{self.selected_kind}: {selected} | {self.status[:150]}",
            14,
            28,
            0.32,
        )
        return canvas

    def _compose(self):
        self.controls = {}
        builder = self._blank(self.BUILDER_H, self.WIDTH, 8)
        cv2.line(
            builder,
            (self.LEFT_W, 0),
            (self.LEFT_W, self.BUILDER_H),
            (52, 52, 60),
            1,
        )
        cv2.line(
            builder,
            (self.LEFT_W + self.CENTER_W, 0),
            (self.LEFT_W + self.CENTER_W, self.BUILDER_H),
            (52, 52, 60),
            1,
        )

        self._tools(builder)
        self._graph(builder)
        self._inspector(builder)

        return np.vstack(
            [
                self._header(),
                builder,
                self._observation(),
                self._footer(),
            ]
        )

    def run(self):
        total = self.HEADER_H + self.BUILDER_H + self.OBS_H + self.FOOTER_H
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.WIDTH, total)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        self.toggle_camera()

        try:
            running = True
            while running:
                now = time.time()
                dt = max(now - self._last, 1e-6)
                self._last = now
                self.fps = 0.9 * self.fps + 0.1 / dt

                if self.camera_on and not self.paused and self.cap is not None:
                    ok, frame = self.cap.read()
                    if ok:
                        self.last_frame = frame.copy()
                        self.last_features = self.extractor.extract(frame)
                        self.last_neural = (
                            self.engine.process(
                                self.last_features,
                                learn=self.learning_on,
                            )
                            if self.engine
                            else None
                        )

                cv2.imshow(self.WINDOW, self._compose())
                running = self._key(cv2.waitKey(1))
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    VisualExperimentUI().run()
