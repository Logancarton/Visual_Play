"""
adaptive_layer.py

Adaptive sparse-plastic processing layer for Visual_Play.

Receives VisionFeatures and learns sparse retinotopic representations without
storing screenshots or explicit templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from vision_features import VisionFeatures


@dataclass
class AdaptiveResult:
    activity: np.ndarray
    normalized_input: np.ndarray
    reconstruction: np.ndarray
    novelty: float
    active_count: int
    feedback_steps: int
    mean_threshold: float
    mean_weight_change: float
    new_routes: int
    pruned_routes: int


class AdaptivePlasticLayer:
    """Sparse recurrent cortical layer with local-style plasticity."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        neurons: int = 256,
        active_fraction: float = 0.08,
        receptive_radius: int = 6,
        norm_rate: float = 0.005,
        learning_rate: float = 0.035,
        recurrent_learning_rate: float = 0.015,
        threshold_rate: float = 0.003,
        target_activity: float = 0.05,
        recurrent_gain: float = 0.40,
        feedback_trigger: float = 0.004,
        max_feedback_steps: int = 5,
        recurrent_initial_degree: int = 12,
        recurrent_candidate_degree: int = 36,
        structural_plasticity: bool = True,
        growth_rate: float = 0.005,
        growth_threshold: float = 0.15,
        growth_decay: float = 0.992,
        prune_threshold: float = 0.002,
        seed: int = 42,
    ) -> None:
        channels, rows, cols = input_shape
        if channels < 1 or rows < 1 or cols < 1:
            raise ValueError("input_shape must be (channels, rows, cols) with positive sizes.")
        if neurons < 2:
            raise ValueError("neurons must be >= 2")

        self.channels = int(channels)
        self.rows = int(rows)
        self.cols = int(cols)
        self.input_dim = self.channels * self.rows * self.cols
        self.neurons = int(neurons)

        self.active_fraction = float(np.clip(active_fraction, 1.0 / neurons, 1.0))
        self.receptive_radius = max(1, int(receptive_radius))

        self.norm_rate = float(np.clip(norm_rate, 1e-7, 1.0))
        self.learning_rate = float(max(0.0, learning_rate))
        self.recurrent_learning_rate = float(max(0.0, recurrent_learning_rate))
        self.threshold_rate = float(max(0.0, threshold_rate))
        self.target_activity = float(np.clip(target_activity, 0.0, 1.0))

        self.recurrent_gain = float(max(0.0, recurrent_gain))
        self.feedback_trigger = float(max(0.0, feedback_trigger))
        self.max_feedback_steps = max(0, int(max_feedback_steps))

        self.structural_plasticity = bool(structural_plasticity)
        self.growth_rate = float(max(0.0, growth_rate))
        self.growth_threshold = float(max(1e-6, growth_threshold))
        self.growth_decay = float(np.clip(growth_decay, 0.0, 1.0))
        self.prune_threshold = float(max(0.0, prune_threshold))

        self.rng = np.random.default_rng(seed)

        self.running_mean = np.zeros(self.input_dim, dtype=np.float32)
        self.running_var = np.ones(self.input_dim, dtype=np.float32)
        self._norm_initialized = False

        self.thresholds = np.full(self.neurons, 0.12, dtype=np.float32)

        self.neuron_centers = []
        self.input_mask = self._build_receptive_field_mask()

        self.W = self.rng.normal(
            loc=0.0,
            scale=0.08,
            size=(self.neurons, self.input_dim),
        ).astype(np.float32)
        self.W *= self.input_mask
        self._normalize_feedforward_weights()

        self.recurrent_candidate_mask = self._build_recurrent_candidate_mask(
            recurrent_candidate_degree
        )
        self.recurrent_mask = self._initial_recurrent_mask(
            recurrent_initial_degree
        )

        self.R = self.rng.normal(
            loc=0.0,
            scale=0.04,
            size=(self.neurons, self.neurons),
        ).astype(np.float32)
        self.R *= self.recurrent_mask
        np.fill_diagonal(self.R, 0.0)

        self.growth_score = np.zeros(
            (self.neurons, self.neurons),
            dtype=np.float32,
        )
        self.previous_activity = np.zeros(self.neurons, dtype=np.float32)

    def _flat_index(self, channel: int, row: int, col: int) -> int:
        return channel * (self.rows * self.cols) + row * self.cols + col

    def _build_receptive_field_mask(self) -> np.ndarray:
        mask = np.zeros((self.neurons, self.input_dim), dtype=np.float32)

        aspect = self.cols / max(self.rows, 1)
        center_cols = max(1, int(np.ceil(np.sqrt(self.neurons * aspect))))
        center_rows = max(1, int(np.ceil(self.neurons / center_cols)))

        centers = []
        for r in range(center_rows):
            rr = int(round(r * (self.rows - 1) / max(center_rows - 1, 1)))
            for c in range(center_cols):
                cc = int(round(c * (self.cols - 1) / max(center_cols - 1, 1)))
                centers.append((rr, cc))

        self.neuron_centers = []
        sigma = max(self.receptive_radius / 1.5, 1e-6)
        for n in range(self.neurons):
            center_r, center_c = centers[n % len(centers)]
            center_r = int(np.clip(center_r + int(self.rng.integers(-1, 2)), 0, self.rows - 1))
            center_c = int(np.clip(center_c + int(self.rng.integers(-1, 2)), 0, self.cols - 1))
            self.neuron_centers.append((center_r, center_c))

            r0 = max(0, center_r - self.receptive_radius)
            r1 = min(self.rows, center_r + self.receptive_radius + 1)
            c0 = max(0, center_c - self.receptive_radius)
            c1 = min(self.cols, center_c + self.receptive_radius + 1)

            for ch in range(self.channels):
                for rr in range(r0, r1):
                    for cc in range(c0, c1):
                        dist_sq = (rr - center_r) ** 2 + (cc - center_c) ** 2
                        mask[n, self._flat_index(ch, rr, cc)] = float(
                            np.exp(-dist_sq / (2.0 * sigma * sigma))
                        )
        return mask

    def _build_recurrent_candidate_mask(self, degree: int) -> np.ndarray:
        degree = int(np.clip(degree, 1, self.neurons - 1))
        mask = np.zeros((self.neurons, self.neurons), dtype=np.float32)

        for target in range(self.neurons):
            choices = np.delete(np.arange(self.neurons), target)
            sources = self.rng.choice(
                choices,
                size=min(degree, len(choices)),
                replace=False,
            )
            mask[target, sources] = 1.0

        np.fill_diagonal(mask, 0.0)
        return mask

    def _initial_recurrent_mask(self, degree: int) -> np.ndarray:
        degree = int(np.clip(degree, 1, self.neurons - 1))
        mask = np.zeros((self.neurons, self.neurons), dtype=np.float32)

        for target in range(self.neurons):
            candidates = np.flatnonzero(self.recurrent_candidate_mask[target] > 0)
            if len(candidates) == 0:
                continue
            chosen = self.rng.choice(
                candidates,
                size=min(degree, len(candidates)),
                replace=False,
            )
            mask[target, chosen] = 1.0

        np.fill_diagonal(mask, 0.0)
        return mask

    def _adaptive_normalize(
        self,
        x: np.ndarray,
        learn: bool,
    ) -> Tuple[np.ndarray, float]:
        x = x.astype(np.float32, copy=False).reshape(-1)
        if x.size != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input signals, got {x.size}.")

        if not self._norm_initialized:
            self.running_mean[:] = x
            self.running_var[:] = 0.04
            self._norm_initialized = True

        std = np.sqrt(self.running_var + 1e-4)
        z = np.clip((x - self.running_mean) / std, -5.0, 5.0)

        novelty = float(np.clip(np.mean(np.abs(z)) / 4.0, 0.0, 1.0))

        if learn:
            delta = x - self.running_mean
            self.running_mean += self.norm_rate * delta
            self.running_var = (
                (1.0 - self.norm_rate) * self.running_var
                + self.norm_rate * (delta * delta)
            )
            self.running_var = np.clip(self.running_var, 1e-4, 4.0)

        return z, novelty

    def _sparsify(self, raw: np.ndarray) -> np.ndarray:
        positive = np.maximum(raw, 0.0)
        k = max(1, int(round(self.neurons * self.active_fraction)))

        if k < self.neurons:
            keep = np.argpartition(positive, -k)[-k:]
            activity = np.zeros_like(positive)
            activity[keep] = positive[keep]
        else:
            activity = positive

        return np.tanh(activity * 1.5).astype(np.float32)

    def _activate(
        self,
        z: np.ndarray,
        recurrent_source: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        drive = self.W @ z
        if recurrent_source is not None and self.recurrent_gain > 0.0:
            drive += self.recurrent_gain * (self.R @ recurrent_source)
        drive -= self.thresholds
        return self._sparsify(drive)

    def _update_feedforward_weights(
        self,
        z: np.ndarray,
        activity: np.ndarray,
        novelty: float,
    ) -> float:
        eta = self.learning_rate * (0.20 + 0.80 * novelty)
        if eta <= 0.0:
            return 0.0

        a_col = activity[:, None]
        delta = eta * (
            a_col * z[None, :]
            - (a_col * a_col) * self.W
        )
        delta *= (self.input_mask > 0.0)

        self.W += delta
        self.W *= (self.input_mask > 0.0)
        np.clip(self.W, -1.2, 1.2, out=self.W)
        return float(np.mean(np.abs(delta)))

    def _update_recurrent_weights(
        self,
        source_activity: np.ndarray,
        target_activity: np.ndarray,
        novelty: float,
    ) -> float:
        if self.recurrent_learning_rate <= 0.0:
            return 0.0

        eta = self.recurrent_learning_rate * (0.20 + 0.80 * novelty)
        hebbian = target_activity[:, None] * source_activity[None, :]
        stabilization = (target_activity[:, None] ** 2) * self.R

        delta = eta * (hebbian - stabilization)
        delta *= self.recurrent_mask

        self.R += delta
        self.R *= self.recurrent_mask
        np.fill_diagonal(self.R, 0.0)
        np.clip(self.R, -0.85, 0.85, out=self.R)
        return float(np.mean(np.abs(delta)))

    def _update_thresholds(self, activity: np.ndarray) -> None:
        self.thresholds += self.threshold_rate * (
            activity - self.target_activity
        )
        np.clip(self.thresholds, -0.10, 1.5, out=self.thresholds)

    def _update_structure(
        self,
        source_activity: np.ndarray,
        target_activity: np.ndarray,
        novelty: float,
    ) -> Tuple[int, int]:
        if not self.structural_plasticity:
            return 0, 0

        self.growth_score *= self.growth_decay
        coactivation = target_activity[:, None] * source_activity[None, :]

        dormant = (
            (self.recurrent_candidate_mask > 0.0)
            & (self.recurrent_mask <= 0.0)
        )
        self.growth_score[dormant] += (
            self.growth_rate
            * (0.25 + 0.75 * novelty)
            * coactivation[dormant]
        )

        grow = dormant & (self.growth_score >= self.growth_threshold)
        new_routes = int(np.count_nonzero(grow))
        if new_routes:
            self.recurrent_mask[grow] = 1.0
            self.R[grow] = 0.02 + 0.05 * np.tanh(self.growth_score[grow])
            self.growth_score[grow] = 0.0

        inactive_pair = coactivation < 1e-4
        decay_mask = (self.recurrent_mask > 0.0) & inactive_pair
        self.R[decay_mask] *= 0.9992

        prune = (
            (self.recurrent_mask > 0.0)
            & (np.abs(self.R) < self.prune_threshold)
        )
        pruned_routes = int(np.count_nonzero(prune))
        if pruned_routes:
            self.recurrent_mask[prune] = 0.0
            self.R[prune] = 0.0

        np.fill_diagonal(self.recurrent_mask, 0.0)
        np.fill_diagonal(self.R, 0.0)
        return new_routes, pruned_routes

    def _normalize_feedforward_weights(self) -> None:
        norms = np.linalg.norm(self.W, axis=1, keepdims=True)
        self.W /= np.maximum(norms, 1e-6)

    def process(
        self,
        features: VisionFeatures,
        learn: bool = True,
    ) -> AdaptiveResult:
        tensor = features.tensor.astype(np.float32, copy=False)
        expected_shape = (self.channels, self.rows, self.cols)
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Adaptive layer expects feature shape {expected_shape}, "
                f"but received {tensor.shape}."
            )

        x = tensor.reshape(-1)
        z, novelty = self._adaptive_normalize(x, learn=learn)

        activity = self._activate(
            z,
            recurrent_source=self.previous_activity,
        )

        feedback_steps = 0
        for _ in range(self.max_feedback_steps):
            feedback = self.recurrent_gain * (self.R @ activity)
            feedback_energy = float(np.mean(np.abs(feedback)))
            if feedback_energy < self.feedback_trigger:
                break

            activity = self._activate(
                z,
                recurrent_source=activity,
            )
            feedback_steps += 1

        mean_weight_change = 0.0
        new_routes = 0
        pruned_routes = 0

        if learn:
            ff_change = self._update_feedforward_weights(
                z,
                activity,
                novelty,
            )
            rec_change = self._update_recurrent_weights(
                self.previous_activity,
                activity,
                novelty,
            )
            mean_weight_change = ff_change + rec_change
            self._update_thresholds(activity)
            new_routes, pruned_routes = self._update_structure(
                self.previous_activity,
                activity,
                novelty,
            )

        self.previous_activity = activity.copy()

        reconstruction = (self.W.T @ activity).reshape(
            self.channels,
            self.rows,
            self.cols,
        )

        return AdaptiveResult(
            activity=activity,
            normalized_input=z,
            reconstruction=reconstruction,
            novelty=novelty,
            active_count=int(np.count_nonzero(activity > 0.0)),
            feedback_steps=feedback_steps,
            mean_threshold=float(np.mean(self.thresholds)),
            mean_weight_change=float(mean_weight_change),
            new_routes=new_routes,
            pruned_routes=pruned_routes,
        )

    def reset_activity(self) -> None:
        self.previous_activity.fill(0.0)

    def reset_expectation(self) -> None:
        self.running_mean.fill(0.0)
        self.running_var.fill(1.0)
        self._norm_initialized = False

    def recurrent_route_count(self) -> int:
        return int(np.count_nonzero(self.recurrent_mask))
