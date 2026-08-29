"""Core spatial neuron-field and synapse substrate for Visual_Play.

This module owns the first real neural state in the project.

It intentionally does not implement learning or structural plasticity yet.
Every neuron has a stable spatial identity, every synapse is an explicit pair,
and propagation is performed through those real pairs rather than through a UI
abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class FieldSnapshot:
    """Read-only copy of one field state after a step."""

    potential: np.ndarray
    activity: np.ndarray


class SpatialNeuronField:
    """A 2D population whose array cells are individual neurons.

    The field keeps stable neuron IDs and normalized XY coordinates. Dynamics
    are deliberately small and explicit:

        potential_t = (1 - leak) * potential_(t-1) + drive
        activity_t  = max(potential_t - threshold, 0) / (1 - threshold)

    Activity is clipped to [0, 1]. This is a graded activation substrate, not a
    claim of a complete biological spiking model.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        threshold: float = 0.20,
        leak: float = 0.35,
        max_potential: float = 2.0,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if not 0.0 <= threshold < 1.0:
            raise ValueError("threshold must be in [0, 1)")
        if not 0.0 <= leak <= 1.0:
            raise ValueError("leak must be in [0, 1]")
        if max_potential <= 0.0:
            raise ValueError("max_potential must be > 0")

        self.rows = int(rows)
        self.cols = int(cols)
        self.neuron_count = self.rows * self.cols
        self.threshold = float(threshold)
        self.leak = float(leak)
        self.max_potential = float(max_potential)

        self.neuron_ids = np.arange(self.neuron_count, dtype=np.int32)

        yy, xx = np.mgrid[0 : self.rows, 0 : self.cols]
        x = xx.astype(np.float32) / max(self.cols - 1, 1)
        y = yy.astype(np.float32) / max(self.rows - 1, 1)
        self.positions = np.stack([x.reshape(-1), y.reshape(-1)], axis=1)

        self.potential = np.zeros(self.neuron_count, dtype=np.float32)
        self.activity = np.zeros(self.neuron_count, dtype=np.float32)

    def _coerce_drive(self, drive: np.ndarray | Iterable[float]) -> np.ndarray:
        array = np.asarray(drive, dtype=np.float32)
        if array.shape == (self.rows, self.cols):
            flat = array.reshape(-1)
        elif array.shape == (self.neuron_count,):
            flat = array
        else:
            raise ValueError(
                f"drive must have shape {(self.rows, self.cols)} or "
                f"({self.neuron_count},), got {array.shape}"
            )
        if not np.all(np.isfinite(flat)):
            raise ValueError("drive must contain only finite values")
        return flat

    def step(self, drive: np.ndarray | Iterable[float]) -> FieldSnapshot:
        """Advance the field one tick from an external/synaptic drive vector."""
        incoming = self._coerce_drive(drive)

        retained = (1.0 - self.leak) * self.potential
        next_potential = retained + incoming
        np.clip(next_potential, 0.0, self.max_potential, out=next_potential)

        denom = max(1.0 - self.threshold, 1e-6)
        next_activity = np.clip(
            (next_potential - self.threshold) / denom,
            0.0,
            1.0,
        ).astype(np.float32)

        self.potential = next_potential.astype(np.float32, copy=False)
        self.activity = next_activity

        return self.snapshot()

    def snapshot(self) -> FieldSnapshot:
        return FieldSnapshot(
            potential=self.potential.copy(),
            activity=self.activity.copy(),
        )

    def activity_map(self) -> np.ndarray:
        return self.activity.reshape(self.rows, self.cols)

    def potential_map(self) -> np.ndarray:
        return self.potential.reshape(self.rows, self.cols)

    def reset(self) -> None:
        self.potential.fill(0.0)
        self.activity.fill(0.0)


class SynapseProjection:
    """Explicit sparse directed synapses between two neuron populations."""

    def __init__(
        self,
        source_count: int,
        target_count: int,
        source_indices: np.ndarray | Iterable[int],
        target_indices: np.ndarray | Iterable[int],
        weights: np.ndarray | Iterable[float],
    ) -> None:
        if source_count < 1 or target_count < 1:
            raise ValueError("source_count and target_count must be >= 1")

        source = np.asarray(source_indices, dtype=np.int32).reshape(-1)
        target = np.asarray(target_indices, dtype=np.int32).reshape(-1)
        weight = np.asarray(weights, dtype=np.float32).reshape(-1)

        if not (source.size == target.size == weight.size):
            raise ValueError("source_indices, target_indices, and weights must match")
        if source.size and (source.min() < 0 or source.max() >= source_count):
            raise ValueError("source index out of range")
        if target.size and (target.min() < 0 or target.max() >= target_count):
            raise ValueError("target index out of range")
        if not np.all(np.isfinite(weight)):
            raise ValueError("weights must contain only finite values")

        pairs = np.stack([source, target], axis=1) if source.size else np.empty((0, 2), dtype=np.int32)
        if pairs.size and np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            raise ValueError("duplicate source-target synapses are not allowed")

        self.source_count = int(source_count)
        self.target_count = int(target_count)
        self.source_indices = source
        self.target_indices = target
        self.weights = weight

    @property
    def synapse_count(self) -> int:
        return int(self.weights.size)

    @classmethod
    def one_to_one(
        cls,
        source_count: int,
        target_count: int,
        *,
        weight: float = 1.0,
    ) -> "SynapseProjection":
        if source_count != target_count:
            raise ValueError("one-to-one projection requires equal population sizes")
        indices = np.arange(source_count, dtype=np.int32)
        weights = np.full(source_count, float(weight), dtype=np.float32)
        return cls(source_count, target_count, indices, indices, weights)

    def propagate(self, source_activity: np.ndarray | Iterable[float]) -> np.ndarray:
        activity = np.asarray(source_activity, dtype=np.float32).reshape(-1)
        if activity.shape != (self.source_count,):
            raise ValueError(
                f"source_activity must have shape ({self.source_count},), "
                f"got {activity.shape}"
            )
        if not np.all(np.isfinite(activity)):
            raise ValueError("source_activity must contain only finite values")

        target_drive = np.zeros(self.target_count, dtype=np.float32)
        contributions = activity[self.source_indices] * self.weights
        np.add.at(target_drive, self.target_indices, contributions)
        return target_drive

    def outgoing(self, source_neuron_id: int) -> Tuple[np.ndarray, np.ndarray]:
        if not 0 <= source_neuron_id < self.source_count:
            raise ValueError("source_neuron_id out of range")
        mask = self.source_indices == int(source_neuron_id)
        return self.target_indices[mask].copy(), self.weights[mask].copy()

    def incoming(self, target_neuron_id: int) -> Tuple[np.ndarray, np.ndarray]:
        if not 0 <= target_neuron_id < self.target_count:
            raise ValueError("target_neuron_id out of range")
        mask = self.target_indices == int(target_neuron_id)
        return self.source_indices[mask].copy(), self.weights[mask].copy()


@dataclass(frozen=True)
class PathwaySnapshot:
    input_activity: np.ndarray
    downstream_activity: np.ndarray


class VisualNeuronPathway:
    """Smallest live Visual_Play signal path.

    Brightness directly drives a spatial input neuron field. A real one-to-one
    synapse projection then drives a second field at the next depth. There is no
    learning rule yet; the purpose is to prove real neuron identity, state, and
    synaptic propagation end to end.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        threshold: float = 0.20,
        leak: float = 0.35,
        projection_weight: float = 0.85,
    ) -> None:
        self.input_field = SpatialNeuronField(
            rows,
            cols,
            threshold=threshold,
            leak=leak,
        )
        self.downstream_field = SpatialNeuronField(
            rows,
            cols,
            threshold=threshold,
            leak=leak,
        )
        self.projection = SynapseProjection.one_to_one(
            self.input_field.neuron_count,
            self.downstream_field.neuron_count,
            weight=projection_weight,
        )

    @property
    def neuron_count(self) -> int:
        return self.input_field.neuron_count + self.downstream_field.neuron_count

    @property
    def synapse_count(self) -> int:
        return self.projection.synapse_count

    def process(self, brightness: np.ndarray) -> PathwaySnapshot:
        first = self.input_field.step(brightness)
        downstream_drive = self.projection.propagate(first.activity)
        second = self.downstream_field.step(downstream_drive)
        return PathwaySnapshot(
            input_activity=first.activity.reshape(
                self.input_field.rows,
                self.input_field.cols,
            ),
            downstream_activity=second.activity.reshape(
                self.downstream_field.rows,
                self.downstream_field.cols,
            ),
        )

    def reset(self) -> None:
        self.input_field.reset()
        self.downstream_field.reset()
