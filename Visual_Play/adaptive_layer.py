"""
adaptive_layer.py

Adaptive sparse-plastic processing layer for Visual_Play.

Purpose
-------
Receive VisionFeatures and let familiarity and cortical representations emerge
from repeated exposure without storing screenshots or static templates.

Core mechanisms
---------------
1. Adaptive normalization:
       mu <- (1-a)mu + a*x
       var <- (1-a)var + a*(x-mu)^2
       z = (x-mu) / sqrt(var + eps)

2. Sparse activation:
       a = f(Wz + R a_prev - theta)
       only the strongest k neurons remain active (Sparse Distributed Representations)

3. Novelty-modulated Oja plasticity:
       dW = eta * novelty * a * (z - aW)

4. Homeostatic thresholds:
       theta <- theta + beta * (activity - target_activity)

5. Recurrent feedback & attractor settling:
       downstream activity triggers micro-cycles until attractor stabilization

6. Structural plasticity:
       synaptogenesis between co-active neurons; pruning of disused routes

7. Cortical Perception Reconstruction:
       x_hat = W^T @ a (visual projection of internal cortical representation)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
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
    """
    Sparse recurrent cortical layer with local-style plasticity.

    The layer learns statistical familiarity and useful pathways over time.
    It never stores a frame or template.
    """

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

        # Running sensory expectation.
        self.running_mean = np.zeros(self.input_dim, dtype=np.float32)
        self.running_var = np.ones(self.input_dim, dtype=np.float32)
        self._norm_initialized = False

        # Adaptive neuronal thresholds.
        self.thresholds = np.full(self.neurons, 0.12, dtype=np.float32)

        # Build spatially local feed-forward receptive fields.
        self.neuron_centers = []
        self.input_mask = self._build_receptive_field_mask()

        self.W = self.rng.normal(
            loc=0.0,
            scale=0.08,
            size=(self.neurons, self.input_dim),
        ).astype(np.float32)
        self.W *= self.input_mask

        # Normalize each neuron's connected weights.
        self._normalize_feedforward_weights()

        # Recurrent pathway candidates and initially active routes.
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

        # Dormant-route co-activation accumulator.
        self.growth_score = np.zeros(
            (self.neurons, self.neurons),
            dtype=np.float32,
        )

        self.previous_activity = np.zeros(self.neurons, dtype=np.float32)

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _flat_index(self, channel: int, row: int, col: int) -> int:
        return channel * (self.rows * self.cols) + row * self.cols + col

    def _build_receptive_field_mask(self) -> np.ndarray:
        mask = np.zeros(
            (self.neurons, self.input_dim),
            dtype=np.float32,
        )

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
        for n in range(self.neurons):
            center_r, center_c = centers[n % len(centers)]

            jr = int(self.rng.integers(-1, 2))
            jc = int(self.rng.integers(-1, 2))
            center_r = int(np.clip(center_r + jr, 0, self.rows - 1))
            center_c = int(np.clip(center_c + jc, 0, self.cols - 1))
            self.neuron_centers.append((center_r, center_c))

            r0 = max(0, center_r - self.receptive_radius)
            r1 = min(self.rows, center_r + self.receptive_radius + 1)
            c0 = max(0, center_c - self.receptive_radius)
            c1 = min(self.cols, center_c + self.receptive_radius + 1)

            for ch in range(self.channels):
                for rr in range(r0, r1):
                    for cc in range(c0, c1):
                        dist_sq = (rr - center_r) ** 2 + (cc - center_c) ** 2
                        weight_falloff = np.exp(-dist_sq / (2.0 * (self.receptive_radius / 1.5) ** 2))
                        mask[n, self._flat_index(ch, rr, cc)] = float(weight_falloff)

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

    # ------------------------------------------------------------------
    # Sensory normalization / novelty
    # ------------------------------------------------------------------

    def _adaptive_normalize(
        self,
        x: np.ndarray,
        learn: bool,
    ) -> Tuple[np.ndarray, float]:
        x = x.astype(np.float32, copy=False).reshape(-1)

        if x.size != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input signals, got {x.size}."
            )

        if not self._norm_initialized:
            self.running_mean[:] = x
            self.running_var[:] = 0.04
            self._norm_initialized = True

        std = np.sqrt(self.running_var + 1e-4)
        z = (x - self.running_mean) / std
        z = np.clip(z, -5.0, 5.0)

        novelty = float(np.mean(np.abs(z)) / 4.0)
        novelty = float(np.clip(novelty, 0.0, 1.0))

        if learn:
            rate = self.norm_rate
            delta = x - self.running_mean
            self.running_mean += rate * delta
            self.running_var = (
                (1.0 - rate) * self.running_var
                + rate * (delta * delta)
            )
            self.running_var = np.clip(self.running_var, 1e-4, 4.0)

        return z, novelty

    # ------------------------------------------------------------------
    # Activation / sparse routing
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Plasticity
    # ------------------------------------------------------------------

    def _update_feedforward_weights(
        self,
        z: np.ndarray,
        activity: np.ndarray,
        novelty: float,
    ) -> float:
        modulation = 0.20 + 0.80 * novelty
        eta = self.learning_rate * modulation

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

        modulation = 0.20 + 0.80 * novelty
        eta = self.recurrent_learning_rate * modulation

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
        norms = np.maximum(norms, 1e-6)
        self.W /= norms

    # ------------------------------------------------------------------
    # Public processing loop
    # ------------------------------------------------------------------

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

        z, novelty = self._adaptive_normalize(
            x,
            learn=learn,
        )

        # Initial feed-forward sensory response.
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

            next_activity = self._activate(
                z,
                recurrent_source=activity,
            )
            activity = next_activity
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

        # Cortical Perceptual Projection: x_hat = W^T @ a
        reconstruction = (self.W.T @ activity).reshape(self.channels, self.rows, self.cols)

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


# ----------------------------------------------------------------------
# Live Cortical Perception HUD with Clickable UI Buttons
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from vision_features import VisionFeatureExtractor

    COLS = 64
    ROWS = 36

    extractor = VisionFeatureExtractor(
        cols=COLS,
        rows=ROWS,
        mirror=True,
        motion_decay=0.70,
        contrast_gain=6.5,
        motion_gain=7.5,
        edge_gain=4.0,
    )

    layer = AdaptivePlasticLayer(
        input_shape=(5, ROWS, COLS),
        neurons=256,
        active_fraction=0.08,
        receptive_radius=6,
        learning_rate=0.035,
        recurrent_learning_rate=0.015,
        recurrent_gain=0.42,
        feedback_trigger=0.003,
        max_feedback_steps=5,
        structural_plasticity=True,
    )

    def open_camera():
        c = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not c.isOpened():
            c = cv2.VideoCapture(0)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return c

    cap = open_camera()
    if cap is None or not cap.isOpened():
        raise SystemExit("Could not access webcam.")

    window = "Visual Play - Cortical Adaptive Engine"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1360, 800)

    # UI Interactive State
    camera_on = True         # Physical camera connection
    learning_on = True       # Learning ON / OFF toggle

    def toggle_camera():
        global cap, camera_on
        if camera_on:
            if cap is not None:
                cap.release()
                cap = None
            camera_on = False
            print(">> CAMERA PHYSICALLY DISCONNECTED (Hardware green light OFF)")
        else:
            cap = open_camera()
            if cap is not None and cap.isOpened():
                camera_on = True
                print(">> CAMERA PHYSICALLY CONNECTED (Hardware green light ON)")
            else:
                camera_on = False
                print(">> Error: Could not re-open camera device.")

    buttons = {}

    def on_mouse(event, x, y, flags, param):
        global learning_on
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn_name, (x1, y1, x2, y2) in buttons.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if btn_name == "power":
                        toggle_camera()
                    elif btn_name == "learning":
                        learning_on = not learning_on
                        print(f"LEARNING: {'[ON]' if learning_on else '[OFF]'}")
                    elif btn_name == "reset_mem":
                        layer.reset_activity()
                        print("Recurrent activity reset.")
                    elif btn_name == "reset_exp":
                        layer.reset_expectation()
                        print("Sensory expectation baseline reset.")
                    elif btn_name == "lr_up":
                        layer.learning_rate = min(0.20, layer.learning_rate + 0.005)
                        print(f"Learning Rate: {layer.learning_rate:.3f}")
                    elif btn_name == "lr_down":
                        layer.learning_rate = max(0.002, layer.learning_rate - 0.005)
                        print(f"Learning Rate: {layer.learning_rate:.3f}")

    cv2.setMouseCallback(window, on_mouse)

    prev_t = time.time()
    fps = 0.0
    last_result: Optional[AdaptiveResult] = None
    last_cam_frame: Optional[np.ndarray] = None

    print("=" * 70)
    print("       VISUAL PLAY - CORTICAL ADAPTIVE PLASTIC LAYER")
    print("=" * 70)
    print("Interactive Controls (Click UI Buttons or use Keyboard):")
    print("  [k], [SPACE], [p] : Physically Connect / Disconnect Camera (Killswitch)")
    print("  [l]               : Toggle Learning (ON / OFF)")
    print("  [ [ ] / [ ] ]     : Adjust Learning Rate")
    print("  [r]               : Reset Recurrent Activity Memory")
    print("  [e]               : Reset Sensory Baseline Expectation")
    print("  [q]               : Quit")
    print("=" * 70)

    try:
        while True:
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            if camera_on and cap is not None:
                ok, frame = cap.read()
                if ok:
                    last_cam_frame = frame.copy()
                    features = extractor.extract(frame)
                    last_result = layer.process(
                        features,
                        learn=learning_on,
                    )
                else:
                    time.sleep(0.01)
                    continue
            else:
                time.sleep(0.03)
        else:
            # Engine paused: read frame to keep buffer fresh or show paused state
            ok, frame = cap.read()
            if ok:
                last_cam_frame = frame.copy()

        # Canvas layout setup
        cam_h, cam_w = 340, 600
        header_h = 50
        total_w = cam_w * 2

        # ----------------- TOP HEADER TOOLBAR -----------------
        header = np.zeros((header_h, total_w, 3), dtype=np.uint8)

        # 1. Power / Camera Killswitch Button
        btn_p_x1, btn_p_y1, btn_p_x2, btn_p_y2 = 15, 8, 200, 42
        buttons["power"] = (btn_p_x1, btn_p_y1, btn_p_x2, btn_p_y2)
        if camera_on:
            cv2.rectangle(header, (btn_p_x1, btn_p_y1), (btn_p_x2, btn_p_y2), (30, 180, 50), -1)
            cv2.rectangle(header, (btn_p_x1, btn_p_y1), (btn_p_x2, btn_p_y2), (80, 255, 120), 2)
            cv2.putText(header, "[CAM: ON - DISCONNECT (k)]", (btn_p_x1 + 8, btn_p_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(header, (btn_p_x1, btn_p_y1), (btn_p_x2, btn_p_y2), (40, 40, 200), -1)
            cv2.rectangle(header, (btn_p_x1, btn_p_y1), (btn_p_x2, btn_p_y2), (80, 80, 255), 2)
            cv2.putText(header, "[CAM: OFF - CONNECT (k)]", (btn_p_x1 + 8, btn_p_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Learning ON / OFF Button
        btn_l_x1, btn_l_y1, btn_l_x2, btn_l_y2 = 215, 8, 380, 42
        buttons["learning"] = (btn_l_x1, btn_l_y1, btn_l_x2, btn_l_y2)
        if learning_on:
            cv2.rectangle(header, (btn_l_x1, btn_l_y1), (btn_l_x2, btn_l_y2), (180, 120, 20), -1)
            cv2.rectangle(header, (btn_l_x1, btn_l_y1), (btn_l_x2, btn_l_y2), (255, 190, 40), 2)
            cv2.putText(header, "[LEARNING: ON]", (btn_l_x1 + 10, btn_l_y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(header, (btn_l_x1, btn_l_y1), (btn_l_x2, btn_l_y2), (50, 50, 60), -1)
            cv2.rectangle(header, (btn_l_x1, btn_l_y1), (btn_l_x2, btn_l_y2), (100, 100, 110), 2)
            cv2.putText(header, "[LEARNING: FROZEN]", (btn_l_x1 + 6, btn_l_y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # 3. Reset Recurrent Button
        btn_r_x1, btn_r_y1, btn_r_x2, btn_r_y2 = 395, 8, 535, 42
        buttons["reset_mem"] = (btn_r_x1, btn_r_y1, btn_r_x2, btn_r_y2)
        cv2.rectangle(header, (btn_r_x1, btn_r_y1), (btn_r_x2, btn_r_y2), (60, 40, 70), -1)
        cv2.rectangle(header, (btn_r_x1, btn_r_y1), (btn_r_x2, btn_r_y2), (160, 100, 200), 1)
        cv2.putText(header, "RESET REC [r]", (btn_r_x1 + 8, btn_r_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 210, 255), 1, cv2.LINE_AA)

        # 4. Reset Baseline Expectation Button
        btn_e_x1, btn_e_y1, btn_e_x2, btn_e_y2 = 545, 8, 685, 42
        buttons["reset_exp"] = (btn_e_x1, btn_e_y1, btn_e_x2, btn_e_y2)
        cv2.rectangle(header, (btn_e_x1, btn_e_y1), (btn_e_x2, btn_e_y2), (60, 50, 30), -1)
        cv2.rectangle(header, (btn_e_x1, btn_e_y1), (btn_e_x2, btn_e_y2), (200, 160, 60), 1)
        cv2.putText(header, "RESET BASE [e]", (btn_e_x1 + 8, btn_e_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 230, 150), 1, cv2.LINE_AA)

        # 5. Learning Rate Buttons [-] / [+]
        btn_lrd_x1, btn_lrd_y1, btn_lrd_x2, btn_lrd_y2 = 700, 8, 750, 42
        buttons["lr_down"] = (btn_lrd_x1, btn_lrd_y1, btn_lrd_x2, btn_lrd_y2)
        cv2.rectangle(header, (btn_lrd_x1, btn_lrd_y1), (btn_lrd_x2, btn_lrd_y2), (40, 40, 50), -1)
        cv2.rectangle(header, (btn_lrd_x1, btn_lrd_y1), (btn_lrd_x2, btn_lrd_y2), (100, 100, 120), 1)
        cv2.putText(header, "LR -", (btn_lrd_x1 + 9, btn_lrd_y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        btn_lru_x1, btn_lru_y1, btn_lru_x2, btn_lru_y2 = 760, 8, 810, 42
        buttons["lr_up"] = (btn_lru_x1, btn_lru_y1, btn_lru_x2, btn_lru_y2)
        cv2.rectangle(header, (btn_lru_x1, btn_lru_y1), (btn_lru_x2, btn_lru_y2), (40, 40, 50), -1)
        cv2.rectangle(header, (btn_lru_x1, btn_lru_y1), (btn_lru_x2, btn_lru_y2), (100, 100, 120), 1)
        cv2.putText(header, "LR +", (btn_lru_x1 + 8, btn_lru_y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # Title & Info text on top right
        status_info = f"ETA: {layer.learning_rate:.3f} | {fps:.1f} FPS"
        cv2.putText(header, status_info, (total_w - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

        # ----------------- MAIN TILES -----------------
        if last_cam_frame is not None and camera_on:
            camera_img = cv2.flip(last_cam_frame, 1)
            cam_tile = cv2.resize(camera_img, (cam_w, cam_h))
        else:
            cam_tile = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        if not camera_on:
            # Standby Dimming Overlay when physically disconnected
            cam_tile = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            cv2.rectangle(cam_tile, (30, cam_h // 2 - 40), (cam_w - 30, cam_h // 2 + 40), (20, 20, 30), -1)
            cv2.rectangle(cam_tile, (30, cam_h // 2 - 40), (cam_w - 30, cam_h // 2 + 40), (60, 60, 220), 2)
            cv2.putText(cam_tile, "CAMERA PHYSICALLY DISCONNECTED", (50, cam_h // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.putText(cam_tile, "Green indicator light OFF | Press [k] or click button to reconnect", (42, cam_h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

            recon_tile = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            cv2.putText(recon_tile, "Cortical Projection Paused", (cam_w // 2 - 130, cam_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (100, 100, 100), 1, cv2.LINE_AA)

            grid_w, grid_h = 400, 240
            act_plasma = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            dash_w = (cam_w * 2) - grid_w
            dash_tile = np.zeros((grid_h, dash_w, 3), dtype=np.uint8)
            cv2.putText(dash_tile, "Hardware Released. Press [k] or Click Button.", (30, grid_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (150, 150, 150), 1, cv2.LINE_AA)

        elif last_result is not None:
            # Overlay active receptive fields on camera
            active_indices = np.flatnonzero(last_result.activity > 0.0)
            for idx in active_indices:
                r_c, c_c = layer.neuron_centers[idx]
                pt_x = int(c_c / COLS * cam_w)
                pt_y = int(r_c / ROWS * cam_h)
                act_val = float(last_result.activity[idx])
                rad = int(8 + act_val * 14)
                cv2.circle(cam_tile, (pt_x, pt_y), rad, (0, 240, 255), 1, cv2.LINE_AA)

            cv2.rectangle(cam_tile, (0, 0), (cam_w, 28), (15, 15, 20), -1)
            cv2.putText(cam_tile, "1. SENSORY RETINA (Webcam + Active Receptive Fields)", (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

            # Cortical Reconstruction (x_hat = W^T @ a)
            recon_lum = last_result.reconstruction[0]
            recon_edges = last_result.reconstruction[3] + last_result.reconstruction[4]
            recon_motion = last_result.reconstruction[2]

            p_r = np.clip(recon_motion * 4.0, 0.0, 1.0)
            p_g = np.clip(recon_edges * 3.0, 0.0, 1.0)
            p_b = np.clip(recon_lum * 2.0, 0.0, 1.0)

            recon_rgb = np.stack([p_b, p_g, p_r], axis=2)
            recon_uint8 = (np.clip(recon_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            recon_tile = cv2.resize(recon_uint8, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)

            cv2.rectangle(recon_tile, (0, 0), (cam_w, 28), (15, 15, 20), -1)
            cv2.putText(recon_tile, "2. CORTICAL RECONSTRUCTION (Internal Neural Perception)", (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 130), 1, cv2.LINE_AA)

            # 2D Cortical Activity Grid
            grid_w, grid_h = 400, 240
            activity_matrix = last_result.activity.reshape(16, 16)
            act_uint8 = (np.clip(activity_matrix, 0.0, 1.0) * 255.0).astype(np.uint8)
            act_plasma = cv2.applyColorMap(
                cv2.resize(act_uint8, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST),
                cv2.COLORMAP_PLASMA,
            )

            cv2.rectangle(act_plasma, (0, 0), (grid_w, 26), (15, 15, 20), -1)
            cv2.putText(act_plasma, f"3. CORTICAL MATRIX ({last_result.active_count}/{layer.neurons} firing)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 100, 255), 1, cv2.LINE_AA)

            # Telemetry Dashboard
            dash_w = (cam_w * 2) - grid_w
            dash_tile = np.zeros((grid_h, dash_w, 3), dtype=np.uint8)

            novelty_pct = last_result.novelty * 100.0
            routes = layer.recurrent_route_count()

            lines = [
                (f"Plasticity Engine: {'[ACTIVE LEARNING]' if learning_on else '[FROZEN]'} (rate: {layer.learning_rate:.3f})", (0, 255, 255) if learning_on else (150, 150, 150)),
                (f"Novelty / Prediction Error: {novelty_pct:5.1f}% | Recurrent Settling: {last_result.feedback_steps} cycles", (50, 220, 255)),
                (f"Active Lateral Routes: {routes:,}  (+{last_result.new_routes} grown / -{last_result.pruned_routes} pruned)", (100, 255, 120)),
                (f"Synaptic Delta (dW): {last_result.mean_weight_change * 1000.0:.4f} mU | Threshold Mean: {last_result.mean_threshold:.3f}", (255, 200, 80)),
                (f"Performance: {fps:.1f} FPS | Resolution: {COLS}x{ROWS}x5 = {COLS*ROWS*5:,} inputs", (200, 200, 200)),
            ]

            bar_w = int((novelty_pct / 100.0) * (dash_w - 40))
            cv2.rectangle(dash_tile, (20, 195), (dash_w - 20, 215), (40, 40, 40), -1)
            bar_color = (40, 100, 255) if novelty_pct > 25 else (0, 240, 200)
            cv2.rectangle(dash_tile, (20, 195), (20 + max(0, min(dash_w - 40, bar_w)), 215), bar_color, -1)
            cv2.putText(dash_tile, f"NOVELTY METER: {novelty_pct:.1f}%", (24, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

            for i, (text, col) in enumerate(lines):
                cv2.putText(dash_tile, text, (20, 32 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

        top_row = np.hstack([cam_tile, recon_tile])
        bottom_row = np.hstack([act_plasma, dash_tile])
        final_layout = np.vstack([header, top_row, bottom_row])

        cv2.imshow(window, final_layout)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("k"), ord(" "), ord("p")):
            toggle_camera()
        elif key == ord("l"):
            learning_on = not learning_on
            print(f"Learning {'ON' if learning_on else 'OFF'}")
        elif key == ord("]"):
            layer.learning_rate = min(0.20, layer.learning_rate + 0.005)
            print(f"Learning Rate increased: {layer.learning_rate:.3f}")
        elif key == ord("["):
            layer.learning_rate = max(0.002, layer.learning_rate - 0.005)
            print(f"Learning Rate decreased: {layer.learning_rate:.3f}")
        elif key == ord("r"):
            layer.reset_activity()
            print("Short-term recurrent activity reset.")
        elif key == ord("e"):
            layer.reset_expectation()
            print("Adaptive expectation reset.")

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(">> Camera released and windows closed.")