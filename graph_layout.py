"""Topology-aware graph layout for the Visual_Play experiment editor.

Layout is presentation support only. It may consume future runtime connection
strengths to visualize learned coupling, but it does not own or create neural
plasticity.
"""

from __future__ import annotations

from collections import defaultdict, deque
from math import hypot
from typing import Dict, Mapping, Tuple

from experiment_spec import ExperimentSpec, SENSORY_SOURCES

Position = Tuple[float, float]


def _layer_depths(spec: ExperimentSpec) -> Dict[str, int]:
    incoming = {layer.id: 0 for layer in spec.layers}
    outgoing = defaultdict(list)
    depth = {layer.id: 0 for layer in spec.layers}

    for connection in spec.connections:
        if connection.source_id in incoming:
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

    # Cyclic islands are not forced into a fake hierarchy. Keep them together in
    # one later band and let dynamic relaxation show their mutual coupling.
    if len(seen) < len(spec.layers):
        fallback = max(depth.values(), default=0) + 1
        for layer in spec.layers:
            if layer.id not in seen:
                depth[layer.id] = fallback

    # A direct sensory input defines the first neural depth.
    for connection in spec.connections:
        if connection.source_id in SENSORY_SOURCES:
            depth[connection.target_id] = min(depth.get(connection.target_id, 0), 0)

    return depth


def grid_positions(spec: ExperimentSpec) -> Dict[str, Position]:
    """Return a deterministic top-down grid with branches spread horizontally."""
    spec.validate()
    if not spec.layers:
        return {}

    depths = _layer_depths(spec)
    groups = defaultdict(list)
    for layer in spec.layers:
        groups[depths[layer.id]].append(layer)

    max_depth = max(groups, default=0)
    result: Dict[str, Position] = {}
    for level in sorted(groups):
        items = sorted(groups[level], key=lambda layer: layer.name.lower())
        count = len(items)
        y = 0.16 + (0.68 * level / max(max_depth, 1)) if max_depth else 0.22
        if max_depth and level == max_depth:
            y = min(y, 0.84)
        for index, layer in enumerate(items):
            x = (index + 1) / (count + 1)
            result[layer.id] = (float(x), float(y))
    return result


def apply_grid(spec: ExperimentSpec, *, unpin: bool = True) -> None:
    positions = grid_positions(spec)
    for layer in spec.layers:
        if unpin:
            layer.pinned = False
        if layer.id in positions:
            layer.x, layer.y = positions[layer.id]


def _sensor_anchor(sensor_id: str) -> Position:
    index = SENSORY_SOURCES.index(sensor_id)
    return ((index + 1.0) / (len(SENSORY_SOURCES) + 1.0), -0.20)


def relaxed_positions(
    spec: ExperimentSpec,
    *,
    runtime_strengths: Mapping[str, float] | None = None,
    step: float = 0.035,
) -> Dict[str, Position]:
    """Compute one cheap spring/repulsion step for unpinned neural fields.

    When a future neural engine returns learned connection strengths, those
    values may be passed as ``runtime_strengths``. Until then, configured gain is
    used only as a visual attraction parameter.
    """
    spec.validate()
    runtime_strengths = runtime_strengths or {}
    positions = {layer.id: (layer.x, layer.y) for layer in spec.layers}
    forces = {layer.id: [0.0, 0.0] for layer in spec.layers}
    lookup = {layer.id: layer for layer in spec.layers}

    # Repel fields that overlap or cluster too tightly.
    ids = list(positions)
    for index, left_id in enumerate(ids):
        for right_id in ids[index + 1 :]:
            lx, ly = positions[left_id]
            rx, ry = positions[right_id]
            dx, dy = rx - lx, ry - ly
            distance = max(hypot(dx, dy), 1e-4)
            target = 0.22
            if distance >= target:
                continue
            magnitude = (target - distance) * 0.55
            ux, uy = dx / distance, dy / distance
            forces[left_id][0] -= ux * magnitude
            forces[left_id][1] -= uy * magnitude
            forces[right_id][0] += ux * magnitude
            forces[right_id][1] += uy * magnitude

    # Connections behave like visual springs. They do not change neural state.
    for connection in spec.connections:
        target = positions.get(connection.target_id)
        if target is None:
            continue
        if connection.source_id in SENSORY_SOURCES:
            source = _sensor_anchor(connection.source_id)
            desired_dy = 0.34
        else:
            source = positions.get(connection.source_id)
            if source is None:
                continue
            desired_dy = 0.26

        sx, sy = source
        tx, ty = target
        configured = max(connection.gain, 0.05)
        strength = max(float(runtime_strengths.get(connection.id, configured)), 0.05)
        coupling = min(2.5, 0.40 + 0.22 * strength)

        dx_error = tx - sx
        dy_error = (ty - sy) - desired_dy
        fx = dx_error * 0.20 * coupling
        fy = dy_error * 0.22 * coupling

        forces[connection.target_id][0] -= fx
        forces[connection.target_id][1] -= fy
        if connection.source_id in forces:
            forces[connection.source_id][0] += fx * 0.65
            forces[connection.source_id][1] += fy * 0.65

    result: Dict[str, Position] = {}
    for layer_id, (x, y) in positions.items():
        layer = lookup[layer_id]
        if layer.pinned:
            result[layer_id] = (x, y)
            continue
        fx, fy = forces[layer_id]
        result[layer_id] = (
            float(max(0.04, min(0.96, x + fx * step))),
            float(max(0.05, min(0.94, y + fy * step))),
        )
    return result


def apply_relaxation(
    spec: ExperimentSpec,
    *,
    runtime_strengths: Mapping[str, float] | None = None,
    step: float = 0.035,
) -> None:
    for layer_id, (x, y) in relaxed_positions(
        spec,
        runtime_strengths=runtime_strengths,
        step=step,
    ).items():
        layer = spec.layer_by_id(layer_id)
        if layer is not None:
            layer.x = x
            layer.y = y
