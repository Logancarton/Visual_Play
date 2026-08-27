"""
parietal_association.py

Dorsal-stream destination for Visual_Play.

Purpose
-------
Receive DORSAL ("where/how") activity from visual_association.py and build
higher-order spatial, motion, and action-relevant representations.

Output Kinematics
-----------------
1. Spatial Centroid: (x, y) normalized coordinate of active motion in visual space.
2. Motion Vector: (vx, vy) dominant directional velocity vector.
3. Looming Factor: radial expansion (>0 approaching, <0 receding).
4. Semantic Action Label: e.g. "FIXATION", "MOTION_LEFT", "LOOMING", etc.
5. Spatial Attention Map: 2D retinotopic heatmap for gaze / motor guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from visual_association import StreamState


@dataclass
class ParietalResult:
    activity: np.ndarray
    motion_trace: np.ndarray
    state_delta: np.ndarray
    active_count: int
    recurrent_steps: int
    mean_threshold: float
    mean_weight_change: float
    motion_energy: float

    # --- Concrete Spatial Kinematics ---
    centroid: Tuple[float, float]          # (x, y) normalized center of motion (0.0 to 1.0)
    motion_vector: Tuple[float, float]     # (vx, vy) velocity vector of visual movement
    looming_factor: float                  # >0 approaching / expanding, <0 receding
    action_label: str                      # High-level semantic action state
    spatial_attention_map: np.ndarray      # 2D normalized attention heatmap (rows, cols)


class ParietalAssociationLayer:
    """
    Higher-order dorsal association layer.

    Extracts spatial coordinates, motion vectors, looming expansion, and action states.
    """

    def __init__(
        self,
        input_dim: int,
        neurons: int = 192,
        grid_shape: Tuple[int, int] = (36, 64),
        active_fraction: float = 0.07,
        trace_decay: float = 0.85,
        delta_gain: float = 0.70,
        learning_rate: float = 0.010,
        recurrent_learning_rate: float = 0.004,
        threshold_rate: float = 0.0015,
        target_activity: float = 0.04,
        recurrent_gain: float = 0.34,
        recurrent_degree: int = 12,
        feedback_trigger: float = 0.0015,
        max_feedback_steps: int = 5,
        seed: int = 301,
    ) -> None:
        self.input_dim = int(input_dim)
        self.neurons = int(neurons)
        self.grid_rows, self.grid_cols = grid_shape

        self.active_fraction = float(np.clip(active_fraction, 1.0 / self.neurons, 1.0))
        self.trace_decay = float(np.clip(trace_decay, 0.0, 0.9999))
        self.delta_gain = float(max(0.0, delta_gain))
        self.learning_rate = float(max(0.0, learning_rate))
        self.recurrent_learning_rate = float(max(0.0, recurrent_learning_rate))
        self.threshold_rate = float(max(0.0, threshold_rate))
        self.target_activity = float(np.clip(target_activity, 0.0, 1.0))
        self.recurrent_gain = float(max(0.0, recurrent_gain))
        self.feedback_trigger = float(max(0.0, feedback_trigger))
        self.max_feedback_steps = max(0, int(max_feedback_steps))

        self.rng = np.random.default_rng(seed)

        self.W = self.rng.normal(0.0, 0.05, size=(self.neurons, self.input_dim)).astype(np.float32)
        self._normalize_rows(self.W)

        self.D = self.rng.normal(0.0, 0.04, size=(self.neurons, self.input_dim)).astype(np.float32)
        self._normalize_rows(self.D)

        recurrent_degree = int(np.clip(recurrent_degree, 1, self.neurons - 1))
        self.recurrent_mask = np.zeros((self.neurons, self.neurons), dtype=np.float32)

        for target in range(self.neurons):
            choices = np.delete(np.arange(self.neurons), target)
            sources = self.rng.choice(choices, size=min(recurrent_degree, len(choices)), replace=False)
            self.recurrent_mask[target, sources] = 1.0

        self.R = self.rng.normal(0.0, 0.025, size=(self.neurons, self.neurons)).astype(np.float32)
        self.R *= self.recurrent_mask
        np.fill_diagonal(self.R, 0.0)

        self.thresholds = np.full(self.neurons, 0.08, dtype=np.float32)

        self.previous_input = np.zeros(self.input_dim, dtype=np.float32)
        self.motion_trace = np.zeros(self.input_dim, dtype=np.float32)
        self.previous_activity = np.zeros(self.neurons, dtype=np.float32)

        # Spatial tracking memory
        self.prev_centroid = (0.5, 0.5)
        self.prev_spread = 0.25
        self.smoothed_vector = (0.0, 0.0)

        # Precompute topographic grid coordinates
        yy, xx = np.mgrid[0:self.grid_rows, 0:self.grid_cols]
        self.norm_x = (xx / max(self.grid_cols - 1, 1)).astype(np.float32)
        self.norm_y = (yy / max(self.grid_rows - 1, 1)).astype(np.float32)

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

        return np.tanh(out * 1.5).astype(np.float32)

    def _activate(
        self,
        x: np.ndarray,
        delta: np.ndarray,
        recurrent_source: Optional[np.ndarray],
    ) -> np.ndarray:
        drive = self.W @ x
        drive += self.delta_gain * (self.D @ delta)

        if recurrent_source is not None:
            drive += self.recurrent_gain * (self.R @ recurrent_source)

        drive -= self.thresholds
        return self._sparsify(drive)

    def _update_weights(
        self,
        x: np.ndarray,
        delta: np.ndarray,
        activity: np.ndarray,
        modulation: float,
    ) -> float:
        eta = self.learning_rate * float(np.clip(modulation, 0.0, 1.0))
        a = activity[:, None]

        dW = eta * (a * x[None, :] - (a * a) * self.W)
        dD = (eta * 0.75) * (a * delta[None, :] - (a * a) * self.D)

        self.W += dW
        self.D += dD

        np.clip(self.W, -1.2, 1.2, out=self.W)
        np.clip(self.D, -1.2, 1.2, out=self.D)

        return float(np.mean(np.abs(dW)) + np.mean(np.abs(dD)))

    def _update_recurrent(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        modulation: float,
    ) -> float:
        eta = self.recurrent_learning_rate * float(np.clip(modulation, 0.0, 1.0))
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
        self.thresholds += self.threshold_rate * (activity - self.target_activity)
        np.clip(self.thresholds, -0.10, 1.4, out=self.thresholds)

    def _compute_kinematics(
        self,
        activity: np.ndarray,
        motion_trace: np.ndarray,
        delta: np.ndarray,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float, str, np.ndarray]:
        """
        Compute spatial centroid, motion direction vector, looming factor, and action state.
        """
        # Project active parietal & dorsal signals back onto 2D retinotopic space
        # Dorsal input size = 128 neurons. Reshape to approximate 8x16 spatial mesh.
        dorsal_energy = np.abs(motion_trace) + np.abs(delta) * 1.5
        if dorsal_energy.size >= 128:
            dorsal_mesh = dorsal_energy[:128].reshape(8, 16)
        else:
            dorsal_mesh = np.resize(dorsal_energy, (8, 16))

        # Upscale to full retinotopic grid
        spatial_map = cv2.resize(dorsal_mesh, (self.grid_cols, self.grid_rows), interpolation=cv2.INTER_CUBIC)
        spatial_map = np.maximum(spatial_map, 0.0)
        total_energy = float(np.sum(spatial_map))

        if total_energy > 1e-4:
            # Normalized centroid
            cx = float(np.sum(spatial_map * self.norm_x) / total_energy)
            cy = float(np.sum(spatial_map * self.norm_y) / total_energy)

            # Spatial spread / dispersion (radius of active motion)
            dx = self.norm_x - cx
            dy = self.norm_y - cy
            spread = float(np.sqrt(np.sum(spatial_map * (dx * dx + dy * dy)) / total_energy))
        else:
            cx, cy = 0.5, 0.5
            spread = 0.25

        # Instantaneous velocity vector of centroid shift
        vx = (cx - self.prev_centroid[0]) * 10.0
        vy = (cy - self.prev_centroid[1]) * 10.0

        # Smooth velocity vector with momentum
        sv_x = 0.65 * self.smoothed_vector[0] + 0.35 * vx
        sv_y = 0.65 * self.smoothed_vector[1] + 0.35 * vy
        self.smoothed_vector = (sv_x, sv_y)

        # Looming factor (expansion rate of spatial motion footprint)
        looming = (spread - self.prev_spread) * 8.0
        looming = float(np.clip(looming, -1.0, 1.0))

        self.prev_centroid = (cx, cy)
        self.prev_spread = spread

        # Classify Semantic Action State
        speed = float(np.sqrt(sv_x * sv_x + sv_y * sv_y))
        motion_mag = float(np.mean(dorsal_energy))

        if motion_mag < 0.003:
            action = "FIXATION / IDLE"
        elif looming > 0.35:
            action = "LOOMING / APPROACHING"
        elif looming < -0.35:
            action = "RECEDING / RETREATING"
        elif speed > 0.08:
            if abs(sv_x) > abs(sv_y):
                action = "SWEEP RIGHT ──►" if sv_x > 0 else "◄── SWEEP LEFT"
            else:
                action = "MOVE DOWN ▼" if sv_y > 0 else "MOVE UP ▲"
        elif motion_mag > 0.02:
            action = "KINETIC GESTURE"
        else:
            action = "MICRO-DRIFT"

        norm_spatial_map = (spatial_map / max(np.max(spatial_map), 1e-4)).astype(np.float32)

        return (cx, cy), (sv_x, sv_y), looming, action, norm_spatial_map

    def process(
        self,
        dorsal_state: StreamState,
        modulation: float = 1.0,
        learn: bool = True,
    ) -> ParietalResult:
        x = np.asarray(dorsal_state.activity, dtype=np.float32).reshape(-1)

        if x.size != self.input_dim:
            raise ValueError(f"Expected dorsal input of size {self.input_dim}, got {x.size}.")

        delta = (x - self.previous_input).astype(np.float32)

        self.motion_trace = (
            self.trace_decay * self.motion_trace
            + (1.0 - self.trace_decay) * delta
        ).astype(np.float32)

        activity = self._activate(
            x,
            self.motion_trace,
            self.previous_activity,
        )

        recurrent_steps = 0

        for _ in range(self.max_feedback_steps):
            next_activity = self._activate(
                x,
                self.motion_trace,
                activity,
            )
            state_change = float(np.mean(np.abs(next_activity - activity)))
            activity = next_activity
            recurrent_steps += 1
            if state_change < self.feedback_trigger:
                break

        mean_weight_change = 0.0

        if learn:
            ff = self._update_weights(x, self.motion_trace, activity, modulation)
            rec = self._update_recurrent(self.previous_activity, activity, modulation)
            self._update_thresholds(activity)
            mean_weight_change = ff + rec

        motion_energy = float(np.mean(np.abs(self.motion_trace)))

        # Extract real spatial kinematics
        centroid, vector, looming, action, att_map = self._compute_kinematics(
            activity, self.motion_trace, delta
        )

        self.previous_input = x.copy()
        self.previous_activity = activity.copy()

        return ParietalResult(
            activity=activity,
            motion_trace=self.motion_trace.copy(),
            state_delta=delta.copy(),
            active_count=int(np.count_nonzero(activity > 0.0)),
            recurrent_steps=recurrent_steps,
            mean_threshold=float(np.mean(self.thresholds)),
            mean_weight_change=float(mean_weight_change),
            motion_energy=motion_energy,
            centroid=centroid,
            motion_vector=vector,
            looming_factor=looming,
            action_label=action,
            spatial_attention_map=att_map,
        )

    def reset_activity(self) -> None:
        self.previous_input.fill(0.0)
        self.motion_trace.fill(0.0)
        self.previous_activity.fill(0.0)
        self.prev_centroid = (0.5, 0.5)
        self.prev_spread = 0.25
        self.smoothed_vector = (0.0, 0.0)