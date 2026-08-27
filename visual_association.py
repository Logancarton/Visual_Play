"""
visual_association.py

Dual-stream visual association layer for Visual_Play.

Purpose
-------
Receive sparse activity from adaptive_layer.py and split it into two
higher-order visual pathways:

    VENTRAL ("what") stream:
        form, contrast, orientation, recurring visual configuration
        -> later candidate input for temporal recognition / familiarity systems

    DORSAL ("where/how") stream:
        motion, spatial change, position / action-relevant visual structure
        -> later candidate input for parietal / action-routing systems

Important
---------
This module does NOT store screenshots or explicit object memories.

Routing is not hard-coded by neuron ID.
Each lower-layer neuron's current learned sensory weights are inspected to
estimate whether that neuron is more driven by:

    form channels  -> ventral routing
    motion/spatial -> dorsal routing

As the lower visual layer changes through plasticity, routing can change too.

Core equations
--------------
Lower cortical activity:
    a1

Feature-based routing gates:
    g_v, g_d

Stream inputs:
    x_v = g_v * a1
    x_d = g_d * a1

Higher associative activity:
    a_v = sparse(f(W_v x_v + R_v a_v_prev - theta_v))
    a_d = sparse(f(W_d x_d + R_d a_d_prev - theta_d))

Local Oja plasticity:
    dW = eta * modulation * a * (x - aW)

Homeostasis:
    theta <- theta + beta * (activity - target)

Architecture
------------
vision_features.py
        |
        v
adaptive_layer.py
        |
        v
visual_association.py
        | \
        |  \
        v   v
     VENTRAL  DORSAL
      "what"  "where/how"
        |          |
        v          v
     temporal    parietal/action
      later         later
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from adaptive_layer import AdaptivePlasticLayer, AdaptiveResult


@dataclass
class StreamState:
    activity: np.ndarray
    active_count: int
    mean_threshold: float
    mean_weight_change: float
    recurrent_steps: int


@dataclass
class AssociationResult:
    ventral: StreamState
    dorsal: StreamState

    # Per-lower-neuron routing strengths.
    ventral_gate: np.ndarray
    dorsal_gate: np.ndarray

    # Routed lower-layer activity actually seen by each stream.
    ventral_input: np.ndarray
    dorsal_input: np.ndarray

    # Useful routing diagnostics.
    mean_ventral_gate: float
    mean_dorsal_gate: float
    strongly_ventral_count: int
    strongly_dorsal_count: int
    shared_count: int


class PlasticAssociationStream:
    """
    One sparse recurrent associative stream.

    It receives already-processed cortical activity, not pixels.
    """

    def __init__(
        self,
        input_dim: int,
        neurons: int = 128,
        active_fraction: float = 0.08,
        learning_rate: float = 0.012,
        recurrent_learning_rate: float = 0.004,
        threshold_rate: float = 0.002,
        target_activity: float = 0.05,
        recurrent_gain: float = 0.30,
        recurrent_degree: int = 10,
        feedback_trigger: float = 0.002,
        max_feedback_steps: int = 4,
        seed: int = 1,
    ) -> None:
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if neurons < 2:
            raise ValueError("neurons must be >= 2")

        self.input_dim = int(input_dim)
        self.neurons = int(neurons)

        self.active_fraction = float(
            np.clip(active_fraction, 1.0 / self.neurons, 1.0)
        )
        self.learning_rate = float(max(0.0, learning_rate))
        self.recurrent_learning_rate = float(max(0.0, recurrent_learning_rate))
        self.threshold_rate = float(max(0.0, threshold_rate))
        self.target_activity = float(np.clip(target_activity, 0.0, 1.0))

        self.recurrent_gain = float(max(0.0, recurrent_gain))
        self.feedback_trigger = float(max(0.0, feedback_trigger))
        self.max_feedback_steps = max(0, int(max_feedback_steps))

        self.rng = np.random.default_rng(seed)

        # Feed-forward weights.
        self.W = self.rng.normal(
            0.0,
            0.06,
            size=(self.neurons, self.input_dim),
        ).astype(np.float32)
        self._normalize_rows(self.W)

        # Sparse recurrent graph.
        recurrent_degree = int(
            np.clip(recurrent_degree, 1, self.neurons - 1)
        )

        self.recurrent_mask = np.zeros(
            (self.neurons, self.neurons),
            dtype=np.float32,
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
            0.0,
            0.025,
            size=(self.neurons, self.neurons),
        ).astype(np.float32)
        self.R *= self.recurrent_mask
        np.fill_diagonal(self.R, 0.0)

        self.thresholds = np.full(
            self.neurons,
            0.10,
            dtype=np.float32,
        )

        self.previous_activity = np.zeros(
            self.neurons,
            dtype=np.float32,
        )

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        matrix /= norms

    def _sparsify(self, raw: np.ndarray) -> np.ndarray:
        positive = np.maximum(raw, 0.0)

        k = max(
            1,
            int(round(self.neurons * self.active_fraction)),
        )

        if k < self.neurons:
            keep = np.argpartition(positive, -k)[-k:]
            activity = np.zeros_like(positive)
            activity[keep] = positive[keep]
        else:
            activity = positive

        return np.tanh(activity * 1.5).astype(np.float32)

    def _activate(
        self,
        x: np.ndarray,
        recurrent_source: Optional[np.ndarray],
    ) -> np.ndarray:
        drive = self.W @ x

        if recurrent_source is not None:
            drive += self.recurrent_gain * (
                self.R @ recurrent_source
            )

        drive -= self.thresholds

        return self._sparsify(drive)

    def _update_feedforward(
        self,
        x: np.ndarray,
        activity: np.ndarray,
        modulation: float,
    ) -> float:
        """
        Oja-style local learning.

            dW = eta * m * a * (x - aW)
        """
        eta = self.learning_rate * float(
            np.clip(modulation, 0.0, 1.0)
        )

        if eta <= 0.0:
            return 0.0

        a = activity[:, None]

        delta = eta * (
            a * x[None, :]
            - (a * a) * self.W
        )

        self.W += delta
        np.clip(self.W, -1.2, 1.2, out=self.W)

        return float(np.mean(np.abs(delta)))

    def _update_recurrent(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        modulation: float,
    ) -> float:
        eta = self.recurrent_learning_rate * float(
            np.clip(modulation, 0.0, 1.0)
        )

        if eta <= 0.0:
            return 0.0

        hebbian = current[:, None] * previous[None, :]
        stabilization = (current[:, None] ** 2) * self.R

        delta = eta * (
            hebbian - stabilization
        )
        delta *= self.recurrent_mask

        self.R += delta
        self.R *= self.recurrent_mask
        np.fill_diagonal(self.R, 0.0)
        np.clip(self.R, -0.8, 0.8, out=self.R)

        return float(np.mean(np.abs(delta)))

    def _update_thresholds(
        self,
        activity: np.ndarray,
    ) -> None:
        self.thresholds += self.threshold_rate * (
            activity - self.target_activity
        )

        np.clip(
            self.thresholds,
            -0.10,
            1.5,
            out=self.thresholds,
        )

    def process(
        self,
        x: np.ndarray,
        plasticity_modulation: float,
        learn: bool = True,
    ) -> StreamState:
        x = np.asarray(
            x,
            dtype=np.float32,
        ).reshape(-1)

        if x.size != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} inputs, got {x.size}."
            )

        activity = self._activate(
            x,
            self.previous_activity,
        )

        recurrent_steps = 0

        for _ in range(self.max_feedback_steps):
            feedback = self.recurrent_gain * (
                self.R @ activity
            )

            feedback_energy = float(
                np.mean(np.abs(feedback))
            )

            if feedback_energy < self.feedback_trigger:
                break

            next_activity = self._activate(
                x,
                activity,
            )

            # Stop if the internal pattern is no longer changing much.
            state_change = float(
                np.mean(
                    np.abs(
                        next_activity - activity
                    )
                )
            )

            activity = next_activity
            recurrent_steps += 1

            if state_change < self.feedback_trigger:
                break

        mean_weight_change = 0.0

        if learn:
            ff = self._update_feedforward(
                x,
                activity,
                plasticity_modulation,
            )

            rec = self._update_recurrent(
                self.previous_activity,
                activity,
                plasticity_modulation,
            )

            mean_weight_change = ff + rec

            self._update_thresholds(
                activity
            )

        self.previous_activity = activity.copy()

        return StreamState(
            activity=activity,
            active_count=int(
                np.count_nonzero(activity > 0.0)
            ),
            mean_threshold=float(
                np.mean(self.thresholds)
            ),
            mean_weight_change=float(
                mean_weight_change
            ),
            recurrent_steps=recurrent_steps,
        )

    def reset_activity(self) -> None:
        self.previous_activity.fill(0.0)


class DualStreamVisualAssociation:
    """
    Split lower visual cortex into ventral and dorsal associative streams.

    Routing is based on what each lower-layer neuron has learned to respond to.
    """

    def __init__(
        self,
        lower_layer: AdaptivePlasticLayer,
        ventral_neurons: int = 128,
        dorsal_neurons: int = 128,
        route_temperature: float = 0.20,
        route_floor: float = 0.05,
        route_smoothing: float = 0.95,
        seed: int = 100,
    ) -> None:
        self.lower = lower_layer
        self.input_dim = lower_layer.neurons

        self.route_temperature = float(
            max(0.01, route_temperature)
        )
        self.route_floor = float(
            np.clip(route_floor, 0.0, 0.45)
        )
        self.route_smoothing = float(
            np.clip(route_smoothing, 0.0, 0.9999)
        )

        self.ventral_stream = PlasticAssociationStream(
            input_dim=self.input_dim,
            neurons=ventral_neurons,
            active_fraction=0.08,
            learning_rate=0.012,
            recurrent_learning_rate=0.004,
            recurrent_gain=0.28,
            seed=seed,
        )

        self.dorsal_stream = PlasticAssociationStream(
            input_dim=self.input_dim,
            neurons=dorsal_neurons,
            active_fraction=0.08,
            learning_rate=0.012,
            recurrent_learning_rate=0.004,
            recurrent_gain=0.32,
            seed=seed + 1,
        )

        # Start with shared routing.
        self.ventral_gate = np.full(
            self.input_dim,
            0.5,
            dtype=np.float32,
        )
        self.dorsal_gate = np.full(
            self.input_dim,
            0.5,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _lower_channel_energy(self) -> np.ndarray:
        """
        Estimate what each lower-layer neuron currently responds to.

        Returns:
            shape = (lower_neurons, sensory_channels)

        Each value is the mean absolute learned feed-forward weight for
        that sensory channel inside that neuron's receptive field.
        """
        W = self.lower.W.reshape(
            self.lower.neurons,
            self.lower.channels,
            self.lower.rows * self.lower.cols,
        )

        mask = self.lower.input_mask.reshape(
            self.lower.neurons,
            self.lower.channels,
            self.lower.rows * self.lower.cols,
        )

        connected = mask > 0.0

        energy = np.zeros(
            (self.lower.neurons, self.lower.channels),
            dtype=np.float32,
        )

        abs_w = np.abs(W)

        for ch in range(self.lower.channels):
            ch_mask = connected[:, ch, :]
            numerator = np.sum(
                abs_w[:, ch, :] * ch_mask,
                axis=1,
            )
            denominator = np.maximum(
                np.sum(ch_mask, axis=1),
                1.0,
            )
            energy[:, ch] = numerator / denominator

        # Normalize each neuron's channel preference.
        total = np.maximum(
            energy.sum(axis=1, keepdims=True),
            1e-8,
        )

        return energy / total

    def _compute_route_gates(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ventral gets preference for:
            brightness, contrast, horizontal edges, vertical edges

        Dorsal gets preference for:
            motion strongly, plus orientation/spatial edges

        Routing remains soft: a neuron can contribute to both pathways.
        """
        pref = self._lower_channel_energy()

        if pref.shape[1] < 5:
            raise ValueError(
                "DualStreamVisualAssociation expects the 5-channel "
                "VisionFeatures layout."
            )

        brightness = pref[:, 0]
        contrast = pref[:, 1]
        motion = pref[:, 2]
        horizontal = pref[:, 3]
        vertical = pref[:, 4]

        # "What" pathway: stable shape/configuration.
        ventral_score = (
            0.15 * brightness
            + 0.30 * contrast
            + 0.275 * horizontal
            + 0.275 * vertical
        )

        # "Where/how" pathway: change and spatial dynamics.
        dorsal_score = (
            0.70 * motion
            + 0.15 * horizontal
            + 0.15 * vertical
        )

        # Two-way softmax.
        stacked = np.stack(
            [ventral_score, dorsal_score],
            axis=1,
        )

        stacked = stacked / self.route_temperature
        stacked -= np.max(
            stacked,
            axis=1,
            keepdims=True,
        )

        exp_s = np.exp(stacked)
        gates = exp_s / np.maximum(
            exp_s.sum(axis=1, keepdims=True),
            1e-8,
        )

        ventral = gates[:, 0]
        dorsal = gates[:, 1]

        # Never completely disconnect a lower neuron from either stream.
        ventral = (
            self.route_floor
            + (1.0 - 2.0 * self.route_floor) * ventral
        )
        dorsal = (
            self.route_floor
            + (1.0 - 2.0 * self.route_floor) * dorsal
        )

        # Smooth routing changes over time.
        s = self.route_smoothing

        self.ventral_gate = (
            s * self.ventral_gate
            + (1.0 - s) * ventral
        ).astype(np.float32)

        self.dorsal_gate = (
            s * self.dorsal_gate
            + (1.0 - s) * dorsal
        ).astype(np.float32)

        return (
            self.ventral_gate,
            self.dorsal_gate,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(
        self,
        lower_result: AdaptiveResult,
        learn: bool = True,
    ) -> AssociationResult:
        lower_activity = np.asarray(
            lower_result.activity,
            dtype=np.float32,
        ).reshape(-1)

        if lower_activity.size != self.input_dim:
            raise ValueError(
                f"Expected lower activity size {self.input_dim}, "
                f"got {lower_activity.size}."
            )

        ventral_gate, dorsal_gate = (
            self._compute_route_gates()
        )

        ventral_input = (
            lower_activity * ventral_gate
        )
        dorsal_input = (
            lower_activity * dorsal_gate
        )

        # Use lower-layer novelty as a broad neuromodulatory signal,
        # but do not make plasticity go to zero for familiar input.
        modulation = float(
            np.clip(
                0.20 + 0.80 * lower_result.novelty,
                0.0,
                1.0,
            )
        )

        ventral_state = self.ventral_stream.process(
            ventral_input,
            plasticity_modulation=modulation,
            learn=learn,
        )

        dorsal_state = self.dorsal_stream.process(
            dorsal_input,
            plasticity_modulation=modulation,
            learn=learn,
        )

        strongly_ventral = int(
            np.count_nonzero(
                ventral_gate >= 0.65
            )
        )
        strongly_dorsal = int(
            np.count_nonzero(
                dorsal_gate >= 0.65
            )
        )
        shared = int(
            np.count_nonzero(
                (ventral_gate > 0.35)
                & (ventral_gate < 0.65)
            )
        )

        return AssociationResult(
            ventral=ventral_state,
            dorsal=dorsal_state,
            ventral_gate=ventral_gate.copy(),
            dorsal_gate=dorsal_gate.copy(),
            ventral_input=ventral_input.copy(),
            dorsal_input=dorsal_input.copy(),
            mean_ventral_gate=float(
                np.mean(ventral_gate)
            ),
            mean_dorsal_gate=float(
                np.mean(dorsal_gate)
            ),
            strongly_ventral_count=strongly_ventral,
            strongly_dorsal_count=strongly_dorsal,
            shared_count=shared,
        )

    def reset_activity(self) -> None:
        self.ventral_stream.reset_activity()
        self.dorsal_stream.reset_activity()


# ----------------------------------------------------------------------
# Live test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import time

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

    lower = AdaptivePlasticLayer(
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

    association = DualStreamVisualAssociation(
        lower_layer=lower,
        ventral_neurons=128,
        dorsal_neurons=128,
    )

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION,
    )

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise SystemExit(
            "Could not access webcam."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )
    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    window = (
        "Visual Play - Dual Stream Association"
    )

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL,
    )
    cv2.resizeWindow(
        window,
        1360,
        800,
    )

    learning_on = True
    previous_time = time.time()
    fps = 0.0

    print("=" * 72)
    print("       VISUAL PLAY - DUAL STREAM VISUAL ASSOCIATION")
    print("=" * 72)
    print("Ventral = form / 'what' pathway")
    print("Dorsal  = motion / spatial / 'where-how' pathway")
    print("")
    print("  [l] Toggle learning")
    print("  [r] Reset recurrent activity")
    print("  [q] Quit")
    print("=" * 72)

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        now = time.time()
        dt = now - previous_time
        previous_time = now

        if dt > 0:
            fps = (
                0.9 * fps
                + 0.1 * (1.0 / dt)
            )

        features = extractor.extract(frame)

        lower_result = lower.process(
            features,
            learn=learning_on,
        )

        result = association.process(
            lower_result,
            learn=learning_on,
        )

        # --------------------------------------------------------------
        # Display
        # --------------------------------------------------------------

        camera = cv2.flip(frame, 1)
        camera = cv2.resize(
            camera,
            (640, 360),
        )

        cv2.rectangle(
            camera,
            (0, 0),
            (640, 30),
            (12, 12, 16),
            -1,
        )
        cv2.putText(
            camera,
            "LOWER VISUAL CORTEX INPUT",
            (12, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 240, 255),
            1,
            cv2.LINE_AA,
        )

        # Lower cortical activity.
        lower_grid = lower_result.activity.reshape(
            16,
            16,
        )

        lower_img = (
            np.clip(
                lower_grid,
                0.0,
                1.0,
            )
            * 255.0
        ).astype(np.uint8)

        lower_img = cv2.applyColorMap(
            cv2.resize(
                lower_img,
                (320, 360),
                interpolation=cv2.INTER_NEAREST,
            ),
            cv2.COLORMAP_PLASMA,
        )

        # Routing preference visualization:
        # green = ventral, orange/red = dorsal.
        route_img = np.zeros(
            (16, 16, 3),
            dtype=np.float32,
        )

        v_gate = result.ventral_gate.reshape(
            16,
            16,
        )
        d_gate = result.dorsal_gate.reshape(
            16,
            16,
        )

        route_img[:, :, 1] = v_gate
        route_img[:, :, 2] = d_gate

        route_img = (
            np.clip(
                route_img,
                0.0,
                1.0,
            )
            * 255.0
        ).astype(np.uint8)

        route_img = cv2.resize(
            route_img,
            (320, 360),
            interpolation=cv2.INTER_NEAREST,
        )

        top = np.hstack(
            [
                camera,
                lower_img,
                route_img,
            ]
        )

        # Ventral and dorsal activity panels.
        ventral_grid = result.ventral.activity.reshape(
            8,
            16,
        )
        dorsal_grid = result.dorsal.activity.reshape(
            8,
            16,
        )

        def stream_panel(
            grid: np.ndarray,
            title: str,
            colormap: int,
        ) -> np.ndarray:
            raw = (
                np.clip(
                    grid,
                    0.0,
                    1.0,
                )
                * 255.0
            ).astype(np.uint8)

            panel = cv2.applyColorMap(
                cv2.resize(
                    raw,
                    (640, 250),
                    interpolation=cv2.INTER_NEAREST,
                ),
                colormap,
            )

            cv2.rectangle(
                panel,
                (0, 0),
                (640, 30),
                (12, 12, 16),
                -1,
            )

            cv2.putText(
                panel,
                title,
                (12, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

            return panel

        ventral_panel = stream_panel(
            ventral_grid,
            (
                f"VENTRAL / WHAT   "
                f"{result.ventral.active_count}/"
                f"{association.ventral_stream.neurons} active"
            ),
            cv2.COLORMAP_VIRIDIS,
        )

        dorsal_panel = stream_panel(
            dorsal_grid,
            (
                f"DORSAL / WHERE-HOW   "
                f"{result.dorsal.active_count}/"
                f"{association.dorsal_stream.neurons} active"
            ),
            cv2.COLORMAP_INFERNO,
        )

        bottom_streams = np.hstack(
            [
                ventral_panel,
                dorsal_panel,
            ]
        )

        status = np.zeros(
            (105, 1280, 3),
            dtype=np.uint8,
        )

        lines = [
            (
                f"Learning: {'ON' if learning_on else 'OFF'}   "
                f"FPS: {fps:.1f}   "
                f"Lower novelty: {lower_result.novelty * 100.0:.1f}%"
            ),
            (
                f"Routing: ventral-strong {result.strongly_ventral_count} | "
                f"dorsal-strong {result.strongly_dorsal_count} | "
                f"shared {result.shared_count}"
            ),
            (
                f"Ventral recurrent cycles: {result.ventral.recurrent_steps} | "
                f"Dorsal recurrent cycles: {result.dorsal.recurrent_steps}"
            ),
        ]

        for i, text in enumerate(lines):
            cv2.putText(
                status,
                text,
                (18, 28 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        final = np.vstack(
            [
                top,
                bottom_streams,
                status,
            ]
        )

        cv2.imshow(
            window,
            final,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("l"):
            learning_on = not learning_on

        elif key == ord("r"):
            lower.reset_activity()
            association.reset_activity()

    cap.release()
    cv2.destroyAllWindows()