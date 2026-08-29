"""Core spatial neuron-field and early retinal signal logic for Visual_Play.

This module owns modeled neuron state and the first live retinal branch.
The live branch is deliberately graded rather than spiking: each visual location
maintains its own adapting luminance baseline, then identical math separates
positive and negative luminance change into ON-like and OFF-like populations.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class FieldSnapshot:
    """Read-only copy of one field state after a step."""

    potential: np.ndarray
    activity: np.ndarray


class SpatialNeuronField:
    """A 2D population whose array cells are individual modeled neurons."""

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

        pairs = (
            np.stack([source, target], axis=1)
            if source.size
            else np.empty((0, 2), dtype=np.int32)
        )
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
class RetinalVariationSnapshot:
    """Observable state of the first graded retinal split."""

    baseline: np.ndarray
    on_activity: np.ndarray
    off_activity: np.ndarray
    timestamp_ms: float


class RetinalVariationPathway:
    """First live retinal-style branching mechanism.

    Each visual location uses identical math in its own spatial position:

        delta = current_luminance - adapting_baseline
        ON    = max(delta, 0) * gain
        OFF   = max(-delta, 0) * gain

    The baseline then moves toward current luminance with a time-based
    exponential update. The first frame seeds the baseline and produces no
    variation response, so the pathway reports change rather than the picture
    itself.

    ON/OFF here means positive/negative luminance variation, not binary firing.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        baseline_tau_ms: float = 350.0,
        response_gain: float = 3.0,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if baseline_tau_ms <= 0.0:
            raise ValueError("baseline_tau_ms must be > 0")
        if response_gain <= 0.0:
            raise ValueError("response_gain must be > 0")

        self.rows = int(rows)
        self.cols = int(cols)
        self.sensory_location_count = self.rows * self.cols
        self.baseline_tau_ms = float(baseline_tau_ms)
        self.response_gain = float(response_gain)

        self.on_field = SpatialNeuronField(
            self.rows,
            self.cols,
            threshold=0.0,
            leak=1.0,
            max_potential=1.0,
        )
        self.off_field = SpatialNeuronField(
            self.rows,
            self.cols,
            threshold=0.0,
            leak=1.0,
            max_potential=1.0,
        )

        self.baseline = np.zeros((self.rows, self.cols), dtype=np.float32)
        self._initialized = False
        self._last_timestamp_ms: float | None = None

    @property
    def neuron_count(self) -> int:
        return self.on_field.neuron_count + self.off_field.neuron_count

    def _coerce_luminance(self, brightness: np.ndarray) -> np.ndarray:
        current = np.asarray(brightness, dtype=np.float32)
        if current.shape != (self.rows, self.cols):
            raise ValueError(
                f"brightness must have shape {(self.rows, self.cols)}, "
                f"got {current.shape}"
            )
        if not np.all(np.isfinite(current)):
            raise ValueError("brightness must contain only finite values")
        return np.clip(current, 0.0, 1.0)

    def process(
        self,
        brightness: np.ndarray,
        *,
        timestamp_ms: float | None = None,
    ) -> RetinalVariationSnapshot:
        current = self._coerce_luminance(brightness)
        now_ms = (
            time.monotonic() * 1000.0
            if timestamp_ms is None
            else float(timestamp_ms)
        )
        if not math.isfinite(now_ms):
            raise ValueError("timestamp_ms must be finite")
        if self._last_timestamp_ms is not None and now_ms < self._last_timestamp_ms:
            raise ValueError("timestamp_ms cannot move backward")

        if not self._initialized:
            self.baseline = current.copy()
            self._initialized = True
            self._last_timestamp_ms = now_ms
            self.on_field.reset()
            self.off_field.reset()
            return self.snapshot(now_ms)

        delta = current - self.baseline
        on_drive = np.clip(delta * self.response_gain, 0.0, 1.0)
        off_drive = np.clip(-delta * self.response_gain, 0.0, 1.0)

        on_snapshot = self.on_field.step(on_drive)
        off_snapshot = self.off_field.step(off_drive)

        dt_ms = now_ms - float(self._last_timestamp_ms)
        alpha = 1.0 - math.exp(-dt_ms / self.baseline_tau_ms) if dt_ms > 0.0 else 0.0
        self.baseline = (self.baseline + alpha * delta).astype(np.float32)
        self._last_timestamp_ms = now_ms

        return RetinalVariationSnapshot(
            baseline=self.baseline.copy(),
            on_activity=on_snapshot.activity.reshape(self.rows, self.cols),
            off_activity=off_snapshot.activity.reshape(self.rows, self.cols),
            timestamp_ms=now_ms,
        )

    def snapshot(self, timestamp_ms: float | None = None) -> RetinalVariationSnapshot:
        now_ms = self._last_timestamp_ms if timestamp_ms is None else float(timestamp_ms)
        if now_ms is None:
            now_ms = 0.0
        return RetinalVariationSnapshot(
            baseline=self.baseline.copy(),
            on_activity=self.on_field.activity_map().copy(),
            off_activity=self.off_field.activity_map().copy(),
            timestamp_ms=float(now_ms),
        )

    def reset(self) -> None:
        self.baseline.fill(0.0)
        self.on_field.reset()
        self.off_field.reset()
        self._initialized = False
        self._last_timestamp_ms = None
