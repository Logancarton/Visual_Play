"""Spatial neuron substrate and live early-retinal signal branches."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class FieldSnapshot:
    potential: np.ndarray
    activity: np.ndarray


class SpatialNeuronField:
    """2D modeled-neuron population with stable IDs and XY positions."""

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
        yy, xx = np.mgrid[0:self.rows, 0:self.cols]
        x = xx.astype(np.float32) / max(self.cols - 1, 1)
        y = yy.astype(np.float32) / max(self.rows - 1, 1)
        self.positions = np.stack((x.ravel(), y.ravel()), axis=1)

        self.potential = np.zeros(self.neuron_count, dtype=np.float32)
        self.activity = np.zeros(self.neuron_count, dtype=np.float32)

    def _coerce_drive(self, drive: np.ndarray | Iterable[float]) -> np.ndarray:
        array = np.asarray(drive, dtype=np.float32)
        if array.shape == (self.rows, self.cols):
            flat = array.ravel()
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
        potential = (1.0 - self.leak) * self.potential + incoming
        potential = np.clip(potential, 0.0, self.max_potential).astype(np.float32)
        denom = max(1.0 - self.threshold, 1e-6)
        activity = np.clip((potential - self.threshold) / denom, 0.0, 1.0)
        self.potential = potential
        self.activity = activity.astype(np.float32)
        return self.snapshot()

    def snapshot(self) -> FieldSnapshot:
        return FieldSnapshot(self.potential.copy(), self.activity.copy())

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

        source = np.asarray(source_indices, dtype=np.int32).ravel()
        target = np.asarray(target_indices, dtype=np.int32).ravel()
        weight = np.asarray(weights, dtype=np.float32).ravel()
        if not (source.size == target.size == weight.size):
            raise ValueError("source_indices, target_indices, and weights must match")
        if source.size and (source.min() < 0 or source.max() >= source_count):
            raise ValueError("source index out of range")
        if target.size and (target.min() < 0 or target.max() >= target_count):
            raise ValueError("target index out of range")
        if not np.all(np.isfinite(weight)):
            raise ValueError("weights must contain only finite values")
        if source.size:
            pairs = np.stack((source, target), axis=1)
            if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
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
        activity = np.asarray(source_activity, dtype=np.float32).ravel()
        if activity.shape != (self.source_count,):
            raise ValueError(
                f"source_activity must have shape ({self.source_count},), "
                f"got {activity.shape}"
            )
        if not np.all(np.isfinite(activity)):
            raise ValueError("source_activity must contain only finite values")
        target_drive = np.zeros(self.target_count, dtype=np.float32)
        np.add.at(
            target_drive,
            self.target_indices,
            activity[self.source_indices] * self.weights,
        )
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
class HorizontalFlowSnapshot:
    rightward_activity: np.ndarray
    leftward_activity: np.ndarray


class HorizontalDirectionalFlowField:
    """Paired left/right opponent comparison over a supplied delayed trace."""

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        flow_gain: float = 4.0,
    ) -> None:
        if rows < 1 or cols < 2:
            raise ValueError("rows must be >= 1 and cols must be >= 2")
        if flow_gain <= 0.0:
            raise ValueError("flow_gain must be > 0")

        self.rows = int(rows)
        self.cols = int(cols)
        self.flow_gain = float(flow_gain)
        kwargs = dict(threshold=0.0, leak=1.0, max_potential=1.0)
        self.rightward_field = SpatialNeuronField(rows, cols, **kwargs)
        self.leftward_field = SpatialNeuronField(rows, cols, **kwargs)

    @property
    def neuron_count(self) -> int:
        return self.rightward_field.neuron_count + self.leftward_field.neuron_count

    def _coerce(self, values: np.ndarray, name: str) -> np.ndarray:
        activity = np.asarray(values, dtype=np.float32)
        if activity.shape != (self.rows, self.cols):
            raise ValueError(
                f"{name} must have shape {(self.rows, self.cols)}, got {activity.shape}"
            )
        if not np.all(np.isfinite(activity)):
            raise ValueError(f"{name} must contain only finite values")
        return np.clip(activity, 0.0, 1.0)

    def process(
        self,
        on_activity: np.ndarray,
        off_activity: np.ndarray,
        *,
        on_trace: np.ndarray,
        off_trace: np.ndarray,
    ) -> HorizontalFlowSnapshot:
        on = self._coerce(on_activity, "on_activity")
        off = self._coerce(off_activity, "off_activity")
        delayed_on = self._coerce(on_trace, "on_trace")
        delayed_off = self._coerce(off_trace, "off_trace")

        right_evidence = (
            delayed_on[:, :-1] * on[:, 1:]
            + delayed_off[:, :-1] * off[:, 1:]
        )
        left_evidence = (
            delayed_on[:, 1:] * on[:, :-1]
            + delayed_off[:, 1:] * off[:, :-1]
        )
        opponent_pair = (right_evidence - left_evidence) * self.flow_gain

        right_drive = np.zeros((self.rows, self.cols), dtype=np.float32)
        left_drive = np.zeros((self.rows, self.cols), dtype=np.float32)
        right_drive[:, 1:] = np.clip(opponent_pair, 0.0, 1.0)
        left_drive[:, :-1] = np.clip(-opponent_pair, 0.0, 1.0)

        right = self.rightward_field.step(right_drive)
        left = self.leftward_field.step(left_drive)

        return HorizontalFlowSnapshot(
            right.activity.reshape(self.rows, self.cols),
            left.activity.reshape(self.rows, self.cols),
        )

    def snapshot(self) -> HorizontalFlowSnapshot:
        return HorizontalFlowSnapshot(
            self.rightward_field.activity_map().copy(),
            self.leftward_field.activity_map().copy(),
        )

    def reset(self) -> None:
        self.rightward_field.reset()
        self.leftward_field.reset()


@dataclass(frozen=True)
class VerticalFlowSnapshot:
    downward_activity: np.ndarray
    upward_activity: np.ndarray


class VerticalDirectionalFlowField:
    """Paired up/down opponent comparison over a supplied delayed trace."""

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        flow_gain: float = 4.0,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if flow_gain <= 0.0:
            raise ValueError("flow_gain must be > 0")

        self.rows = int(rows)
        self.cols = int(cols)
        self.flow_gain = float(flow_gain)
        kwargs = dict(threshold=0.0, leak=1.0, max_potential=1.0)
        self.downward_field = SpatialNeuronField(rows, cols, **kwargs)
        self.upward_field = SpatialNeuronField(rows, cols, **kwargs)

    @property
    def neuron_count(self) -> int:
        return self.downward_field.neuron_count + self.upward_field.neuron_count

    def _coerce(self, values: np.ndarray, name: str) -> np.ndarray:
        activity = np.asarray(values, dtype=np.float32)
        if activity.shape != (self.rows, self.cols):
            raise ValueError(
                f"{name} must have shape {(self.rows, self.cols)}, got {activity.shape}"
            )
        if not np.all(np.isfinite(activity)):
            raise ValueError(f"{name} must contain only finite values")
        return np.clip(activity, 0.0, 1.0)

    def process(
        self,
        on_activity: np.ndarray,
        off_activity: np.ndarray,
        *,
        on_trace: np.ndarray,
        off_trace: np.ndarray,
    ) -> VerticalFlowSnapshot:
        on = self._coerce(on_activity, "on_activity")
        off = self._coerce(off_activity, "off_activity")
        delayed_on = self._coerce(on_trace, "on_trace")
        delayed_off = self._coerce(off_trace, "off_trace")

        down_evidence = (
            delayed_on[:-1, :] * on[1:, :]
            + delayed_off[:-1, :] * off[1:, :]
        )
        up_evidence = (
            delayed_on[1:, :] * on[:-1, :]
            + delayed_off[1:, :] * off[:-1, :]
        )
        opponent_pair = (down_evidence - up_evidence) * self.flow_gain

        down_drive = np.zeros((self.rows, self.cols), dtype=np.float32)
        up_drive = np.zeros((self.rows, self.cols), dtype=np.float32)
        down_drive[1:, :] = np.clip(opponent_pair, 0.0, 1.0)
        up_drive[:-1, :] = np.clip(-opponent_pair, 0.0, 1.0)

        down = self.downward_field.step(down_drive)
        up = self.upward_field.step(up_drive)

        return VerticalFlowSnapshot(
            down.activity.reshape(self.rows, self.cols),
            up.activity.reshape(self.rows, self.cols),
        )

    def snapshot(self) -> VerticalFlowSnapshot:
        return VerticalFlowSnapshot(
            self.downward_field.activity_map().copy(),
            self.upward_field.activity_map().copy(),
        )

    def reset(self) -> None:
        self.downward_field.reset()
        self.upward_field.reset()


@dataclass(frozen=True)
class CardinalFlowSnapshot:
    rightward_activity: np.ndarray
    leftward_activity: np.ndarray
    downward_activity: np.ndarray
    upward_activity: np.ndarray
    on_trace: np.ndarray
    off_trace: np.ndarray
    timestamp_ms: float


class CardinalDirectionalFlowField:
    """Four cardinal comparisons sharing one positive/negative trace state."""

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        trace_tau_ms: float = 100.0,
        flow_gain: float = 4.0,
    ) -> None:
        if rows < 1 or cols < 2:
            raise ValueError("rows must be >= 1 and cols must be >= 2")
        if trace_tau_ms <= 0.0 or flow_gain <= 0.0:
            raise ValueError("trace_tau_ms and flow_gain must be > 0")

        self.rows = int(rows)
        self.cols = int(cols)
        self.trace_tau_ms = float(trace_tau_ms)
        self.horizontal_flow = HorizontalDirectionalFlowField(
            rows, cols, flow_gain=flow_gain
        )
        self.vertical_flow = VerticalDirectionalFlowField(
            rows, cols, flow_gain=flow_gain
        )
        self.on_trace = np.zeros((rows, cols), dtype=np.float32)
        self.off_trace = np.zeros((rows, cols), dtype=np.float32)
        self._last_timestamp_ms: float | None = None

    @property
    def neuron_count(self) -> int:
        return self.horizontal_flow.neuron_count + self.vertical_flow.neuron_count

    def _coerce(self, values: np.ndarray, name: str) -> np.ndarray:
        activity = np.asarray(values, dtype=np.float32)
        if activity.shape != (self.rows, self.cols):
            raise ValueError(
                f"{name} must have shape {(self.rows, self.cols)}, got {activity.shape}"
            )
        if not np.all(np.isfinite(activity)):
            raise ValueError(f"{name} must contain only finite values")
        return np.clip(activity, 0.0, 1.0)

    def process(
        self,
        on_activity: np.ndarray,
        off_activity: np.ndarray,
        *,
        timestamp_ms: float,
    ) -> CardinalFlowSnapshot:
        on = self._coerce(on_activity, "on_activity")
        off = self._coerce(off_activity, "off_activity")
        now_ms = float(timestamp_ms)
        if not math.isfinite(now_ms):
            raise ValueError("timestamp_ms must be finite")
        if self._last_timestamp_ms is not None and now_ms < self._last_timestamp_ms:
            raise ValueError("timestamp_ms cannot move backward")

        horizontal = self.horizontal_flow.process(
            on,
            off,
            on_trace=self.on_trace,
            off_trace=self.off_trace,
        )
        vertical = self.vertical_flow.process(
            on,
            off,
            on_trace=self.on_trace,
            off_trace=self.off_trace,
        )

        if self._last_timestamp_ms is None:
            self.on_trace = on.copy()
            self.off_trace = off.copy()
        else:
            dt_ms = now_ms - self._last_timestamp_ms
            alpha = (
                1.0 - math.exp(-dt_ms / self.trace_tau_ms)
                if dt_ms > 0.0
                else 0.0
            )
            self.on_trace += alpha * (on - self.on_trace)
            self.off_trace += alpha * (off - self.off_trace)
        self._last_timestamp_ms = now_ms

        return CardinalFlowSnapshot(
            horizontal.rightward_activity,
            horizontal.leftward_activity,
            vertical.downward_activity,
            vertical.upward_activity,
            self.on_trace.copy(),
            self.off_trace.copy(),
            now_ms,
        )

    def snapshot(self, timestamp_ms: float | None = None) -> CardinalFlowSnapshot:
        now_ms = self._last_timestamp_ms if timestamp_ms is None else float(timestamp_ms)
        horizontal = self.horizontal_flow.snapshot()
        vertical = self.vertical_flow.snapshot()
        return CardinalFlowSnapshot(
            horizontal.rightward_activity,
            horizontal.leftward_activity,
            vertical.downward_activity,
            vertical.upward_activity,
            self.on_trace.copy(),
            self.off_trace.copy(),
            0.0 if now_ms is None else float(now_ms),
        )

    def reset(self) -> None:
        self.horizontal_flow.reset()
        self.vertical_flow.reset()
        self.on_trace.fill(0.0)
        self.off_trace.fill(0.0)
        self._last_timestamp_ms = None


@dataclass(frozen=True)
class RetinalSignalSnapshot:
    baseline: np.ndarray
    on_activity: np.ndarray
    off_activity: np.ndarray
    contrast_activity: np.ndarray
    rightward_activity: np.ndarray
    leftward_activity: np.ndarray
    downward_activity: np.ndarray
    upward_activity: np.ndarray
    timestamp_ms: float


class RetinalSignalPathway:
    """Live brightness -> variation, contrast, and paired directional flow."""

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        baseline_tau_ms: float = 350.0,
        response_gain: float = 3.0,
        contrast_gain: float = 4.0,
        flow_trace_tau_ms: float = 100.0,
        flow_gain: float = 4.0,
    ) -> None:
        if rows < 1 or cols < 2:
            raise ValueError("rows must be >= 1 and cols must be >= 2")
        if min(
            baseline_tau_ms,
            response_gain,
            contrast_gain,
            flow_trace_tau_ms,
            flow_gain,
        ) <= 0.0:
            raise ValueError("time constants and gains must be > 0")

        self.rows = int(rows)
        self.cols = int(cols)
        self.sensory_location_count = self.rows * self.cols
        self.baseline_tau_ms = float(baseline_tau_ms)
        self.response_gain = float(response_gain)
        self.contrast_gain = float(contrast_gain)

        kwargs = dict(threshold=0.0, leak=1.0, max_potential=1.0)
        self.on_field = SpatialNeuronField(rows, cols, **kwargs)
        self.off_field = SpatialNeuronField(rows, cols, **kwargs)
        self.contrast_field = SpatialNeuronField(rows, cols, **kwargs)
        self.directional_flow = CardinalDirectionalFlowField(
            rows,
            cols,
            trace_tau_ms=flow_trace_tau_ms,
            flow_gain=flow_gain,
        )
        self.baseline = np.zeros((rows, cols), dtype=np.float32)
        self._initialized = False
        self._last_timestamp_ms: float | None = None

    @property
    def neuron_count(self) -> int:
        return (
            self.on_field.neuron_count
            + self.off_field.neuron_count
            + self.contrast_field.neuron_count
            + self.directional_flow.neuron_count
        )

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

    @staticmethod
    def _local_surround(current: np.ndarray) -> np.ndarray:
        padded = np.pad(current, 1, mode="edge")
        surround = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        )
        return (surround / 8.0).astype(np.float32)

    def process(
        self,
        brightness: np.ndarray,
        *,
        timestamp_ms: float | None = None,
    ) -> RetinalSignalSnapshot:
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

        surround = self._local_surround(current)
        contrast = self.contrast_field.step(
            np.clip(np.abs(current - surround) * self.contrast_gain, 0.0, 1.0)
        )

        if not self._initialized:
            self.baseline = current.copy()
            self._initialized = True
            self._last_timestamp_ms = now_ms
            self.on_field.reset()
            self.off_field.reset()
            flow = self.directional_flow.process(
                self.on_field.activity_map(),
                self.off_field.activity_map(),
                timestamp_ms=now_ms,
            )
            return RetinalSignalSnapshot(
                self.baseline.copy(),
                self.on_field.activity_map().copy(),
                self.off_field.activity_map().copy(),
                contrast.activity.reshape(self.rows, self.cols),
                flow.rightward_activity,
                flow.leftward_activity,
                flow.downward_activity,
                flow.upward_activity,
                now_ms,
            )

        delta = current - self.baseline
        on = self.on_field.step(np.clip(delta * self.response_gain, 0.0, 1.0))
        off = self.off_field.step(np.clip(-delta * self.response_gain, 0.0, 1.0))
        on_map = on.activity.reshape(self.rows, self.cols)
        off_map = off.activity.reshape(self.rows, self.cols)
        flow = self.directional_flow.process(
            on_map, off_map, timestamp_ms=now_ms
        )

        dt_ms = now_ms - self._last_timestamp_ms
        alpha = (
            1.0 - math.exp(-dt_ms / self.baseline_tau_ms)
            if dt_ms > 0.0
            else 0.0
        )
        self.baseline = (self.baseline + alpha * delta).astype(np.float32)
        self._last_timestamp_ms = now_ms

        return RetinalSignalSnapshot(
            self.baseline.copy(),
            on_map,
            off_map,
            contrast.activity.reshape(self.rows, self.cols),
            flow.rightward_activity,
            flow.leftward_activity,
            flow.downward_activity,
            flow.upward_activity,
            now_ms,
        )

    def snapshot(self, timestamp_ms: float | None = None) -> RetinalSignalSnapshot:
        now_ms = self._last_timestamp_ms if timestamp_ms is None else float(timestamp_ms)
        flow = self.directional_flow.snapshot(now_ms)
        return RetinalSignalSnapshot(
            self.baseline.copy(),
            self.on_field.activity_map().copy(),
            self.off_field.activity_map().copy(),
            self.contrast_field.activity_map().copy(),
            flow.rightward_activity,
            flow.leftward_activity,
            flow.downward_activity,
            flow.upward_activity,
            0.0 if now_ms is None else float(now_ms),
        )

    def reset(self) -> None:
        self.baseline.fill(0.0)
        self.on_field.reset()
        self.off_field.reset()
        self.contrast_field.reset()
        self.directional_flow.reset()
        self._initialized = False
        self._last_timestamp_ms = None
