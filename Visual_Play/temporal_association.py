"""
temporal_association.py

Ventral-stream destination for Visual_Play.

Purpose
-------
Receive the VENTRAL ("what") activity from visual_association.py and build
higher-order, slowly changing form / identity-like representations.

This is NOT explicit object recognition and does NOT store screenshots.

Core mechanisms
---------------
1. Short temporal trace:
       trace_t = decay * trace_(t-1) + (1-decay) * x_t

2. Associative activation:
       a_t = sparse(f(Wx_t + T*trace_t + R*a_(t-1) - theta))

3. Local Oja plasticity:
       dW = eta * modulation * a * (x - aW)

4. Homeostatic thresholds:
       theta <- theta + beta * (activity - target)

5. Slow prototype stabilization:
       repeated ventral patterns gradually recruit similar cell assemblies

Architecture
------------
vision_features.py
        |
adaptive_layer.py
        |
visual_association.py
        |
        +--> ventral.activity
                |
                v
      temporal_association.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from visual_association import StreamState


@dataclass
class TemporalResult:
    activity: np.ndarray
    temporal_trace: np.ndarray
    active_count: int
    recurrent_steps: int
    mean_threshold: float
    mean_weight_change: float
    stability: float


class TemporalAssociationLayer:
    """
    Higher-order ventral association layer.

    Repeated form/configuration patterns should gradually recruit more stable
    distributed activity without storing explicit templates.
    """

    def __init__(
        self,
        input_dim: int,
        neurons: int = 192,
        active_fraction: float = 0.07,
        trace_decay: float = 0.92,
        trace_gain: float = 0.45,
        learning_rate: float = 0.008,
        recurrent_learning_rate: float = 0.003,
        threshold_rate: float = 0.0015,
        target_activity: float = 0.04,
        recurrent_gain: float = 0.30,
        recurrent_degree: int = 12,
        feedback_trigger: float = 0.0015,
        max_feedback_steps: int = 5,
        seed: int = 201,
    ) -> None:
        self.input_dim = int(input_dim)
        self.neurons = int(neurons)

        self.active_fraction = float(
            np.clip(active_fraction, 1.0 / self.neurons, 1.0)
        )
        self.trace_decay = float(np.clip(trace_decay, 0.0, 0.9999))
        self.trace_gain = float(max(0.0, trace_gain))
        self.learning_rate = float(max(0.0, learning_rate))
        self.recurrent_learning_rate = float(max(0.0, recurrent_learning_rate))
        self.threshold_rate = float(max(0.0, threshold_rate))
        self.target_activity = float(np.clip(target_activity, 0.0, 1.0))
        self.recurrent_gain = float(max(0.0, recurrent_gain))
        self.feedback_trigger = float(max(0.0, feedback_trigger))
        self.max_feedback_steps = max(0, int(max_feedback_steps))

        self.rng = np.random.default_rng(seed)

        self.W = self.rng.normal(
            0.0, 0.05, size=(self.neurons, self.input_dim)
        ).astype(np.float32)
        self._normalize_rows(self.W)

        recurrent_degree = int(
            np.clip(recurrent_degree, 1, self.neurons - 1)
        )

        self.recurrent_mask = np.zeros(
            (self.neurons, self.neurons), dtype=np.float32
        )

        for target in range(self.neurons):
            choices = np.delete(np.arange(self.neurons), target)
            sources = self.rng.choice(
                choices,
                size=min(recurrent_degree, len(choices)),
                replace=False,
            )
            self.recurrent_mask[target, sources] = 1.0

        self.R = self.rng.normal(
            0.0, 0.02, size=(self.neurons, self.neurons)
        ).astype(np.float32)
        self.R *= self.recurrent_mask
        np.fill_diagonal(self.R, 0.0)

        # Separate trace weights allow sustained form/context influence.
        self.T = self.rng.normal(
            0.0, 0.03, size=(self.neurons, self.input_dim)
        ).astype(np.float32)
        self._normalize_rows(self.T)

        self.thresholds = np.full(
            self.neurons, 0.08, dtype=np.float32
        )

        self.temporal_trace = np.zeros(
            self.input_dim, dtype=np.float32
        )
        self.previous_activity = np.zeros(
            self.neurons, dtype=np.float32
        )

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.maximum(norms, 1e-6)

    def _sparsify(self, raw: np.ndarray) -> np.ndarray:
        positive = np.maximum(raw, 0.0)
        k = max(1, int(round(self.neurons * self.active_fraction)))

        if k < self.neurons:
            keep = np.argpartition(positive, -k)[-k:]
            out = np.zeros_like(positive)
            out[keep] = positive[keep]
        else:
            out = positive

        return np.tanh(out * 1.4).astype(np.float32)

    def _activate(
        self,
        x: np.ndarray,
        recurrent_source: Optional[np.ndarray],
    ) -> np.ndarray:
        drive = self.W @ x
        drive += self.trace_gain * (self.T @ self.temporal_trace)

        if recurrent_source is not None:
            drive += self.recurrent_gain * (self.R @ recurrent_source)

        drive -= self.thresholds
        return self._sparsify(drive)

    def _update_weights(
        self,
        x: np.ndarray,
        activity: np.ndarray,
        modulation: float,
    ) -> float:
        eta = self.learning_rate * float(np.clip(modulation, 0.0, 1.0))
        a = activity[:, None]

        dW = eta * (
            a * x[None, :]
            - (a * a) * self.W
        )

        self.W += dW
        np.clip(self.W, -1.2, 1.2, out=self.W)

        # Trace association learns more slowly.
        dT = (eta * 0.35) * (
            a * self.temporal_trace[None, :]
            - (a * a) * self.T
        )

        self.T += dT
        np.clip(self.T, -1.0, 1.0, out=self.T)

        return float(np.mean(np.abs(dW)) + np.mean(np.abs(dT)))

    def _update_recurrent(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        modulation: float,
    ) -> float:
        eta = self.recurrent_learning_rate * float(
            np.clip(modulation, 0.0, 1.0)
        )

        hebbian = current[:, None] * previous[None, :]
        stabilization = (current[:, None] ** 2) * self.R

        dR = eta * (hebbian - stabilization)
        dR *= self.recurrent_mask

        self.R += dR
        self.R *= self.recurrent_mask
        np.fill_diagonal(self.R, 0.0)
        np.clip(self.R, -0.8, 0.8, out=self.R)

        return float(np.mean(np.abs(dR)))

    def _update_thresholds(self, activity: np.ndarray) -> None:
        self.thresholds += self.threshold_rate * (
            activity - self.target_activity
        )
        np.clip(self.thresholds, -0.10, 1.4, out=self.thresholds)

    def process(
        self,
        ventral_state: StreamState,
        modulation: float = 1.0,
        learn: bool = True,
    ) -> TemporalResult:
        x = np.asarray(
            ventral_state.activity, dtype=np.float32
        ).reshape(-1)

        if x.size != self.input_dim:
            raise ValueError(
                f"Expected ventral input of size {self.input_dim}, got {x.size}."
            )

        # Short temporal memory of recent ventral structure.
        self.temporal_trace = (
            self.trace_decay * self.temporal_trace
            + (1.0 - self.trace_decay) * x
        ).astype(np.float32)

        activity = self._activate(
            x, self.previous_activity
        )

        recurrent_steps = 0

        for _ in range(self.max_feedback_steps):
            next_activity = self._activate(
                x, activity
            )

            state_change = float(
                np.mean(np.abs(next_activity - activity))
            )

            activity = next_activity
            recurrent_steps += 1

            if state_change < self.feedback_trigger:
                break

        mean_weight_change = 0.0

        if learn:
            ff = self._update_weights(
                x, activity, modulation
            )
            rec = self._update_recurrent(
                self.previous_activity,
                activity,
                modulation,
            )
            self._update_thresholds(activity)
            mean_weight_change = ff + rec

        # Familiar / stable representation = little change from previous state.
        stability = float(
            1.0 - np.clip(
                np.mean(np.abs(activity - self.previous_activity)),
                0.0,
                1.0,
            )
        )

        self.previous_activity = activity.copy()

        return TemporalResult(
            activity=activity,
            temporal_trace=self.temporal_trace.copy(),
            active_count=int(np.count_nonzero(activity > 0.0)),
            recurrent_steps=recurrent_steps,
            mean_threshold=float(np.mean(self.thresholds)),
            mean_weight_change=float(mean_weight_change),
            stability=stability,
        )

    def reset_activity(self) -> None:
        self.previous_activity.fill(0.0)
        self.temporal_trace.fill(0.0)