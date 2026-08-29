"""Structural 3D layout helpers for the Visual_Play experiment editor.

A field is a two-dimensional neuron sheet placed at a structural Z depth.
Ordinary downstream fields remain centered on the same X/Y lane and advance
along Z. Only multiple fields that occupy the same structural depth are split
laterally for branch visibility.

This module never uses synaptic strength or plasticity to move whole fields.
Future learned motion belongs to individual neuron positions returned by the
neural engine.
"""

from __future__ import annotations

from collections import defaultdict, deque
from math import cos, radians, sin
from typing import Dict, Mapping, Tuple

import numpy as np

from experiment_spec import ExperimentSpec, SENSORY_SOURCES

Position2D = Tuple[float, float]
Position3D = Tuple[float, float, float]


def field_depths(spec: ExperimentSpec) -> Dict[str, int]:
    """Compute structural feed-forward depth while containing recurrent cycles."""
    spec.validate()
    incoming = {layer.id: 0 for layer in spec.layers}
    outgoing = defaultdict(list)
    depth = {layer.id: 0 for layer in spec.layers}

    for connection in spec.connections:
        if connection.source_id in incoming and connection.target_id in incoming:
            outgoing[connection.source_id].append(connection.target_id)
            incoming[connection.target_id] += 1

    queue = deque(sorted(layer_id for layer_id, count in incoming.items() if count == 0))
    seen = set()

    while queue:
        current = queue.popleft()
        seen.add(current)
        for target in outgoing[current]:
            depth[target] = max(depth[target], depth[current] + 1)
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)

    # A cyclic island is a peer group, not an infinitely deep chain.
    if len(seen) < len(spec.layers):
        fallback = max((depth[layer_id] for layer_id in seen), default=-1) + 1
        for layer in spec.layers:
            if layer.id not in seen:
                depth[layer.id] = fallback

    # Any field directly driven by a sensory source is the first neural depth.
    for connection in spec.connections:
        if connection.source_id in SENSORY_SOURCES:
            depth[connection.target_id] = 0

    return depth


def field_world_positions(
    spec: ExperimentSpec,
    *,
    depth_spacing: float = 1.8,
    branch_spacing: float = 3.8,
) -> Dict[str, Position3D]:
    """Place neuron fields in 3D structural space.

    X = branch lane.
    Y = zero-centered field plane.
    Z = feed-forward depth.

    A single field at a depth is centered at X=0. Multiple peers at the same
    depth are spread symmetrically on X. Therefore a normal chain advances only
    in Z, while explicit branching becomes lateral.
    """
    spec.validate()
    if not spec.layers:
        return {}

    depths = field_depths(spec)
    groups = defaultdict(list)
    for layer in spec.layers:
        groups[depths[layer.id]].append(layer)

    result: Dict[str, Position3D] = {}
    for level, layers in sorted(groups.items()):
        ordered = sorted(layers, key=lambda layer: (layer.name.lower(), layer.id))
        count = len(ordered)
        for index, layer in enumerate(ordered):
            lane = index - (count - 1) / 2.0
            result[layer.id] = (
                float(lane * branch_spacing),
                0.0,
                float(level * depth_spacing),
            )
    return result


def plane_node_positions(
    rows: int,
    cols: int,
    center: Position3D,
    *,
    width: float = 3.6,
    height: float = 2.4,
) -> np.ndarray:
    """Return one XYZ position per neuron in a rectangular field plane."""
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    cx, cy, cz = map(float, center)

    xs = np.linspace(-width / 2.0, width / 2.0, cols, dtype=np.float32)
    ys = np.linspace(-height / 2.0, height / 2.0, rows, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    return np.column_stack(
        [
            xx.ravel() + cx,
            yy.ravel() + cy,
            np.full(rows * cols, cz, dtype=np.float32),
        ]
    ).astype(np.float32)


def plane_corners(
    center: Position3D,
    *,
    width: float = 3.6,
    height: float = 2.4,
) -> np.ndarray:
    """Return four XYZ corners for a neuron field plane."""
    cx, cy, cz = map(float, center)
    hw = width / 2.0
    hh = height / 2.0
    return np.asarray(
        [
            [cx - hw, cy - hh, cz],
            [cx + hw, cy - hh, cz],
            [cx + hw, cy + hh, cz],
            [cx - hw, cy + hh, cz],
        ],
        dtype=np.float32,
    )


def project_points_3d(
    points: np.ndarray,
    *,
    screen_center: Position2D,
    yaw_degrees: float = 32.0,
    pitch_degrees: float = -16.0,
    camera_distance: float = 8.0,
    focal_length: float = 520.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project XYZ points into screen XY with a cheap perspective camera.

    Returns:
        screen_xy: int32 pixel positions.
        camera_depth: positive per-point distance used for draw ordering.
    """
    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if p.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

    yaw = radians(float(yaw_degrees))
    pitch = radians(float(pitch_degrees))
    cyaw, syaw = cos(yaw), sin(yaw)
    cpitch, spitch = cos(pitch), sin(pitch)

    x1 = cyaw * p[:, 0] + syaw * p[:, 2]
    z1 = -syaw * p[:, 0] + cyaw * p[:, 2]

    y2 = cpitch * p[:, 1] - spitch * z1
    z2 = spitch * p[:, 1] + cpitch * z1

    camera_depth = np.maximum(float(camera_distance) + z2, 0.25)
    scale = float(focal_length) / camera_depth

    cx, cy = map(float, screen_center)
    sx = cx + x1 * scale
    sy = cy - y2 * scale

    return np.column_stack([sx, sy]).astype(np.int32), camera_depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Backward-compatible 2D layout API.
# These remain structural only. Runtime strength can never move whole fields.
# ---------------------------------------------------------------------------

def grid_positions(spec: ExperimentSpec) -> Dict[str, Position2D]:
    spec.validate()
    depths = field_depths(spec)
    max_depth = max(depths.values(), default=0)
    groups = defaultdict(list)
    for layer in spec.layers:
        groups[depths[layer.id]].append(layer)

    result: Dict[str, Position2D] = {}
    for level, layers in sorted(groups.items()):
        ordered = sorted(layers, key=lambda layer: (layer.name.lower(), layer.id))
        count = len(ordered)
        y = 0.16 if max_depth == 0 else 0.12 + 0.76 * (level / max_depth)
        for index, layer in enumerate(ordered):
            x = (index + 1) / (count + 1)
            result[layer.id] = (float(x), float(y))
    return result


def apply_grid(spec: ExperimentSpec, *, unpin: bool = True) -> Dict[str, Position2D]:
    positions = grid_positions(spec)
    for layer in spec.layers:
        if hasattr(layer, "pinned") and unpin:
            layer.pinned = False
        if layer.id in positions and hasattr(layer, "x") and hasattr(layer, "y"):
            layer.x, layer.y = positions[layer.id]
    return positions


def relaxed_positions(
    spec: ExperimentSpec,
    *,
    runtime_strengths: Mapping[str, float] | None = None,
    step: float = 0.035,
) -> Dict[str, Position2D]:
    del runtime_strengths, step
    structural = grid_positions(spec)
    result: Dict[str, Position2D] = {}
    for layer in spec.layers:
        if bool(getattr(layer, "pinned", False)) and hasattr(layer, "x") and hasattr(layer, "y"):
            result[layer.id] = (float(layer.x), float(layer.y))
        else:
            result[layer.id] = structural[layer.id]
    return result


def apply_relaxation(
    spec: ExperimentSpec,
    *,
    runtime_strengths: Mapping[str, float] | None = None,
    step: float = 0.035,
) -> Dict[str, Position2D]:
    positions = relaxed_positions(
        spec,
        runtime_strengths=runtime_strengths,
        step=step,
    )
    for layer in spec.layers:
        if layer.id in positions and hasattr(layer, "x") and hasattr(layer, "y"):
            layer.x, layer.y = positions[layer.id]
    return positions
