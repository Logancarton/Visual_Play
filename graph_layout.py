"""Structural field layout for the Visual_Play experiment editor.

A layer/field card represents a structural depth/population in the experiment.
Its placement is derived from graph depth and branching only.

Plasticity, synaptic strength, and physical pull belong to individual neurons
and synapses inside/between fields. Runtime connection strength must never move
a whole field card.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Mapping, Tuple

from experiment_spec import ExperimentSpec, SENSORY_SOURCES

Position = Tuple[float, float]


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

    # A recurrent/cyclic island is a peer group, not an endlessly deeper chain.
    if len(seen) < len(spec.layers):
        fallback = max((depth[layer_id] for layer_id in seen), default=-1) + 1
        for layer in spec.layers:
            if layer.id not in seen:
                depth[layer.id] = fallback

    for connection in spec.connections:
        if connection.source_id in SENSORY_SOURCES:
            depth[connection.target_id] = 0

    return depth


def grid_positions(spec: ExperimentSpec) -> Dict[str, Position]:
    """Deterministic depth grid: deeper fields down, branches spread sideways."""
    spec.validate()
    if not spec.layers:
        return {}

    depths = field_depths(spec)
    groups = defaultdict(list)
    for layer in spec.layers:
        groups[depths[layer.id]].append(layer)

    max_depth = max(groups, default=0)
    result: Dict[str, Position] = {}
    for level in sorted(groups):
        items = sorted(groups[level], key=lambda layer: (layer.name.lower(), layer.id))
        count = len(items)
        y = 0.16 if max_depth == 0 else 0.12 + 0.76 * (level / max_depth)
        for index, layer in enumerate(items):
            x = (index + 1) / (count + 1)
            result[layer.id] = (float(x), float(y))
    return result


def apply_grid(spec: ExperimentSpec, *, unpin: bool = True) -> Dict[str, Position]:
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
) -> Dict[str, Position]:
    """Compatibility replacement for the old whole-field spring layout.

    ``runtime_strengths`` and ``step`` are deliberately ignored. Whole fields
    are not particles. A manually pinned field may keep its presentation
    position; every automatic field uses the deterministic structural depth grid.
    """
    del runtime_strengths, step
    structural = grid_positions(spec)
    result: Dict[str, Position] = {}
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
) -> Dict[str, Position]:
    """Apply structural positions only; never strength-based whole-field motion."""
    positions = relaxed_positions(
        spec,
        runtime_strengths=runtime_strengths,
        step=step,
    )
    for layer in spec.layers:
        if layer.id in positions and hasattr(layer, "x") and hasattr(layer, "y"):
            layer.x, layer.y = positions[layer.id]
    return positions
