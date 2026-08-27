"""
cortical_system.py

Unified Full Cortical Cognitive Engine for Visual_Play.

Orchestrates the entire 5-layer biological visual architecture:
1. Retina / LGN (vision_features.py): 5-channel sensory feature extraction
2. Primary Cortex V1/V2 (adaptive_layer.py): Sparse retinotopic plastic layer (256 neurons)
3. Dual-Stream Routing (visual_association.py): Learned channel-energy gating into Ventral & Dorsal streams
4. Temporal Association (temporal_association.py): Ventral "What" apex (192 neurons, form memory & stability)
5. Parietal Association (parietal_association.py): Dorsal "Where/How" apex (192 neurons, spatial kinematics, centroid, flow & action)
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from adaptive_layer import AdaptivePlasticLayer
from parietal_association import ParietalAssociationLayer
from temporal_association import TemporalAssociationLayer
from vision_features import VisionFeatureExtractor
from visual_association import DualStreamVisualAssociation


class CorticalSystem:
    """
    Unified biological vision architecture uniting all cortical layers.
    """

    def __init__(
        self,
        cols: int = 64,
        rows: int = 36,
        v1_neurons: int = 256,
        stream_neurons: int = 128,
        apex_neurons: int = 192,
        seed: int = 42,
    ) -> None:
        self.cols = cols
        self.rows = rows

        # 1. Retina / LGN
        self.extractor = VisionFeatureExtractor(
            cols=cols,
            rows=rows,
            mirror=True,
            motion_decay=0.70,
            contrast_gain=6.5,
            motion_gain=7.5,
            edge_gain=4.0,
        )

        # 2. Primary Cortex (V1/V2)
        self.v1 = AdaptivePlasticLayer(
            input_shape=(5, rows, cols),
            neurons=v1_neurons,
            active_fraction=0.08,
            receptive_radius=6,
            learning_rate=0.035,
            recurrent_learning_rate=0.015,
            recurrent_gain=0.42,
            feedback_trigger=0.003,
            max_feedback_steps=5,
            structural_plasticity=True,
            seed=seed,
        )

        # 3. Dual Stream Association (Ventral / Dorsal Split)
        self.assoc = DualStreamVisualAssociation(
            lower_layer=self.v1,
            ventral_neurons=stream_neurons,
            dorsal_neurons=stream_neurons,
            seed=seed + 1,
        )

        # 4. Temporal Association (Ventral "What" Apex)
        self.temporal = TemporalAssociationLayer(
            input_dim=stream_neurons,
            neurons=apex_neurons,
            active_fraction=0.07,
            trace_decay=0.92,
            learning_rate=0.010,
            seed=seed + 2,
        )

        # 5. Parietal Association (Dorsal "Where/How" Apex)
        self.parietal = ParietalAssociationLayer(
            input_dim=stream_neurons,
            neurons=apex_neurons,
            grid_shape=(rows, cols),
            active_fraction=0.07,
            trace_decay=0.85,
            learning_rate=0.012,
            seed=seed + 3,
        )

    def process(self, frame: np.ndarray, learn: bool = True):
        # 1. Extract sensory features
        features = self.extractor.extract(frame)

        # 2. Primary cortex sparse coding & novelty
        v1_res = self.v1.process(features, learn=learn)

        # 3. Route into dual streams
        assoc_res = self.assoc.process(v1_res, learn=learn)

        # Broad neuromodulation gated by lower novelty
        modulation = float(np.clip(0.20 + 0.80 * v1_res.novelty, 0.0, 1.0))

        # 4. Temporal association (Ventral form processing)
        temp_res = self.temporal.process(
            assoc_res.ventral,
            modulation=modulation,
            learn=learn,
        )

        # 5. Parietal association (Dorsal motion & spatial kinematics)
        pari_res = self.parietal.process(
            assoc_res.dorsal,
            modulation=modulation,
            learn=learn,
        )

        return {
            "v1": v1_res,
            "assoc": assoc_res,
            "temporal": temp_res,
            "parietal": pari_res,
        }

    def reset_activity(self) -> None:
        self.v1.reset_activity()
        self.assoc.reset_activity()
        self.temporal.reset_activity()
        self.parietal.reset_activity()

    def reset_expectation(self) -> None:
        self.v1.reset_expectation()


# ----------------------------------------------------------------------
# Full Interactive Cortical Cockpit
# ----------------------------------------------------------------------

if __name__ == "__main__":
    system = CorticalSystem(cols=64, rows=36)

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

    window = "Visual Play - Full Cortical Cockpit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1400, 860)

    # UI Interactive State
    camera_on = True
    learning_on = True

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
                        print(f"LEARNING: {'[ON]' if learning_on else '[FROZEN]'}")
                    elif btn_name == "reset_mem":
                        system.reset_activity()
                        print("All cortical recurrent states reset.")
                    elif btn_name == "reset_exp":
                        system.reset_expectation()
                        print("Sensory baseline expectation reset.")
                    elif btn_name == "lr_up":
                        system.v1.learning_rate = min(0.20, system.v1.learning_rate + 0.005)
                        print(f"V1 Learning Rate: {system.v1.learning_rate:.3f}")
                    elif btn_name == "lr_down":
                        system.v1.learning_rate = max(0.002, system.v1.learning_rate - 0.005)
                        print(f"V1 Learning Rate: {system.v1.learning_rate:.3f}")

    cv2.setMouseCallback(window, on_mouse)

    prev_t = time.time()
    fps = 0.0
    last_res = None
    last_frame = None

    print("=" * 75)
    print("       VISUAL PLAY - FULL CORTICAL ARCHITECTURE COCKPIT")
    print("=" * 75)
    print("Uniting Retina, V1, Dual-Stream Association, Temporal, & Parietal Cortices")
    print("")
    print("Controls:")
    print("  [k], [SPACE], [p] : Physically Connect / Disconnect Camera (Killswitch)")
    print("  [l]            : Toggle Learning (ON / FROZEN)")
    print("  [ [ ] / [ ] ]  : Adjust Learning Rate")
    print("  [r]            : Reset Recurrent Activity Memory")
    print("  [e]            : Reset Baseline Sensory Expectation")
    print("  [q]            : Quit")
    print("=" * 75)

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
                    last_frame = frame.copy()
                    last_res = system.process(frame, learn=learning_on)
                else:
                    time.sleep(0.01)
                    continue
            else:
                time.sleep(0.03)

            total_w = 1380
            header_h = 50

            # --- 1. TOP HEADER TOOLBAR ---
            header = np.zeros((header_h, total_w, 3), dtype=np.uint8)

            # Power / Camera Hardware Killswitch button
            buttons["power"] = (15, 8, 200, 42)
            if camera_on:
                cv2.rectangle(header, (15, 8), (200, 42), (30, 180, 50), -1)
                cv2.rectangle(header, (15, 8), (200, 42), (80, 255, 120), 2)
                cv2.putText(header, "[CAM: ON - DISCONNECT (k)]", (22, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(header, (15, 8), (200, 42), (40, 40, 200), -1)
                cv2.rectangle(header, (15, 8), (200, 42), (80, 80, 255), 2)
                cv2.putText(header, "[CAM: OFF - CONNECT (k)]", (22, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # Learning button
            buttons["learning"] = (215, 8, 390, 42)
            if learning_on:
                cv2.rectangle(header, (215, 8), (390, 42), (180, 120, 20), -1)
                cv2.rectangle(header, (215, 8), (390, 42), (255, 190, 40), 2)
                cv2.putText(header, "[LEARNING: ON]", (225, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(header, (215, 8), (390, 42), (50, 50, 60), -1)
                cv2.rectangle(header, (215, 8), (390, 42), (100, 100, 110), 2)
                cv2.putText(header, "[LEARNING: FROZEN]", (221, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

            # Reset Recurrent
            buttons["reset_mem"] = (405, 8, 545, 42)
            cv2.rectangle(header, (405, 8), (545, 42), (60, 40, 70), -1)
            cv2.rectangle(header, (405, 8), (545, 42), (160, 100, 200), 1)
            cv2.putText(header, "RESET MEM [r]", (415, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 210, 255), 1, cv2.LINE_AA)

            # Reset Expectation
            buttons["reset_exp"] = (555, 8, 700, 42)
            cv2.rectangle(header, (555, 8), (700, 42), (60, 50, 30), -1)
            cv2.rectangle(header, (555, 8), (700, 42), (200, 160, 60), 1)
            cv2.putText(header, "RESET BASE [e]", (563, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 230, 150), 1, cv2.LINE_AA)

            # Learning Rate +/-
            buttons["lr_down"] = (710, 8, 760, 42)
            cv2.rectangle(header, (710, 8), (760, 42), (40, 40, 50), -1)
            cv2.rectangle(header, (710, 8), (760, 42), (100, 100, 120), 1)
            cv2.putText(header, "LR -", (718, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

            buttons["lr_up"] = (770, 8, 820, 42)
            cv2.rectangle(header, (770, 8), (820, 42), (40, 40, 50), -1)
            cv2.rectangle(header, (770, 8), (820, 42), (100, 100, 120), 1)
            cv2.putText(header, "LR +", (778, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

            status_header = f"V1 Rate: {system.v1.learning_rate:.3f} | {fps:.1f} FPS"
            cv2.putText(header, status_header, (total_w - 230, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

            # --- 2. ROW 1: SENSORY RETINA & V1 CORTICAL RECONSTRUCTION ---
            r1_h, r1_w = 280, total_w // 2

            if last_frame is not None and camera_on:
                camera_img = cv2.flip(last_frame, 1)
                cam_tile = cv2.resize(camera_img, (r1_w, r1_h))
            else:
                cam_tile = np.zeros((r1_h, r1_w, 3), dtype=np.uint8)

            if not camera_on:
                cam_tile = np.zeros((r1_h, r1_w, 3), dtype=np.uint8)
                cv2.rectangle(cam_tile, (30, r1_h // 2 - 35), (r1_w - 30, r1_h // 2 + 35), (20, 20, 30), -1)
                cv2.rectangle(cam_tile, (30, r1_h // 2 - 35), (r1_w - 30, r1_h // 2 + 35), (60, 60, 220), 2)
                cv2.putText(cam_tile, "CAMERA PHYSICALLY DISCONNECTED", (50, r1_h // 2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 2, cv2.LINE_AA)
                cv2.putText(cam_tile, "Green indicator light OFF | Press [k] or click button to reconnect", (45, r1_h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
                recon_tile = np.zeros((r1_h, r1_w, 3), dtype=np.uint8)
            elif last_res is not None:
                v1_res = last_res["v1"]
                pari_res = last_res["parietal"]

                # Draw Receptive Field Highlights
                active_indices = np.flatnonzero(v1_res.activity > 0.0)
                for idx in active_indices:
                    r_c, c_c = system.v1.neuron_centers[idx]
                    pt_x = int(c_c / system.cols * r1_w)
                    pt_y = int(r_c / system.rows * r1_h)
                    rad = int(8 + float(v1_res.activity[idx]) * 14)
                    cv2.circle(cam_tile, (pt_x, pt_y), rad, (0, 240, 255), 1, cv2.LINE_AA)

                # Draw Parietal Spatial Motion Tracking Reticle & Vector
                cx, cy = pari_res.centroid
                vx, vy = pari_res.motion_vector
                pt_cx = int(np.clip(cx * r1_w, 10, r1_w - 10))
                pt_cy = int(np.clip(cy * r1_h, 10, r1_h - 10))

                if pari_res.motion_energy > 0.002:
                    cv2.drawMarker(cam_tile, (pt_cx, pt_cy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
                    cv2.circle(cam_tile, (pt_cx, pt_cy), 18, (40, 100, 255), 2, cv2.LINE_AA)

                    end_x = int(np.clip(pt_cx + vx * 25.0, 5, r1_w - 5))
                    end_y = int(np.clip(pt_cy + vy * 25.0, 5, r1_h - 5))
                    cv2.arrowedLine(cam_tile, (pt_cx, pt_cy), (end_x, end_y), (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.35)

                # V1 Cortical Reconstruction
                recon_lum = v1_res.reconstruction[0]
                recon_edges = v1_res.reconstruction[3] + v1_res.reconstruction[4]
                recon_motion = v1_res.reconstruction[2]
                p_r = np.clip(recon_motion * 4.0, 0.0, 1.0)
                p_g = np.clip(recon_edges * 3.0, 0.0, 1.0)
                p_b = np.clip(recon_lum * 2.0, 0.0, 1.0)
                recon_rgb = np.stack([p_b, p_g, p_r], axis=2)
                recon_tile = cv2.resize((np.clip(recon_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), (r1_w, r1_h), interpolation=cv2.INTER_NEAREST)
            else:
                recon_tile = np.zeros((r1_h, r1_w, 3), dtype=np.uint8)

            cv2.rectangle(cam_tile, (0, 0), (r1_w, 26), (15, 15, 20), -1)
            cv2.putText(cam_tile, "1. RETINAL SENSORY INPUT (Gaze Reticle + Receptive Fields)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.rectangle(recon_tile, (0, 0), (r1_w, 26), (15, 15, 20), -1)
            cv2.putText(recon_tile, "2. V1 PERCEPTUAL RECONSTRUCTION (Internal Neural Map: x_hat = W^T a)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 130), 1, cv2.LINE_AA)

            row1 = np.hstack([cam_tile, recon_tile])

            # --- 3. ROW 2: DUAL-STREAM CORTICAL DIVERGENCE (VENTRAL vs DORSAL) ---
            r2_h = 240
            w_panel = total_w // 2

            if camera_on and last_res is not None:
                assoc_res = last_res["assoc"]
                temp_res = last_res["temporal"]
                pari_res = last_res["parietal"]

                # Ventral "What" Panel
                ventral_act = temp_res.activity.reshape(12, 16)
                ventral_img = (np.clip(ventral_act, 0.0, 1.0) * 255.0).astype(np.uint8)
                ventral_color = cv2.applyColorMap(cv2.resize(ventral_img, (w_panel, r2_h), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_VIRIDIS)

                cv2.rectangle(ventral_color, (0, 0), (w_panel, 26), (15, 15, 20), -1)
                cv2.putText(ventral_color, f"3. VENTRAL 'WHAT' APEX (IT Cortex: {temp_res.active_count}/{system.temporal.neurons} active) | Stability: {temp_res.stability*100:.1f}%", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 120), 1, cv2.LINE_AA)

                # Dorsal "Where/How" Panel
                dorsal_act = pari_res.activity.reshape(12, 16)
                dorsal_img = (np.clip(dorsal_act, 0.0, 1.0) * 255.0).astype(np.uint8)
                dorsal_color = cv2.applyColorMap(cv2.resize(dorsal_img, (w_panel, r2_h), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO)

                cv2.rectangle(dorsal_color, (0, 0), (w_panel, 26), (15, 15, 20), -1)
                action_text = f"4. DORSAL 'WHERE/HOW' [ACTION: {pari_res.action_label}] | Kinetic: {pari_res.motion_energy*1000.0:.2f} mU"
                cv2.putText(dorsal_color, action_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

                kinematics_subtext = f"Centroid: ({pari_res.centroid[0]:.2f}, {pari_res.centroid[1]:.2f}) | Vector: ({pari_res.motion_vector[0]:+.2f}, {pari_res.motion_vector[1]:+.2f}) | Looming: {pari_res.looming_factor:+.2f}"
                cv2.rectangle(dorsal_color, (0, r2_h - 24), (w_panel, r2_h), (10, 10, 15), -1)
                cv2.putText(dorsal_color, kinematics_subtext, (10, r2_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
            else:
                ventral_color = np.zeros((r2_h, w_panel, 3), dtype=np.uint8)
                dorsal_color = np.zeros((r2_h, w_panel, 3), dtype=np.uint8)

            row2 = np.hstack([ventral_color, dorsal_color])

            # --- 4. ROW 3: ROUTING MATRIX & TELEMETRY DASHBOARD ---
            r3_h = 160
            r_matrix_w = 260
            dash_w = total_w - r_matrix_w

            if camera_on and last_res is not None:
                v_gate = last_res["assoc"].ventral_gate.reshape(16, 16)
                d_gate = last_res["assoc"].dorsal_gate.reshape(16, 16)
                route_map = np.zeros((16, 16, 3), dtype=np.float32)
                route_map[:, :, 1] = v_gate
                route_map[:, :, 2] = d_gate
                route_uint8 = (np.clip(route_map, 0.0, 1.0) * 255.0).astype(np.uint8)
                route_tile = cv2.resize(route_uint8, (r_matrix_w, r3_h), interpolation=cv2.INTER_NEAREST)
            else:
                route_tile = np.zeros((r3_h, r_matrix_w, 3), dtype=np.uint8)

            cv2.rectangle(route_tile, (0, 0), (r_matrix_w, 24), (15, 15, 20), -1)
            cv2.putText(route_tile, "V1 ROUTE GATES [Grn=V | Red=D]", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

            dash_tile = np.zeros((r3_h, dash_w, 3), dtype=np.uint8)

            if camera_on and last_res is not None:
                v1_res = last_res["v1"]
                assoc_res = last_res["assoc"]
                temp_res = last_res["temporal"]
                pari_res = last_res["parietal"]

                novelty_pct = v1_res.novelty * 100.0
                routes = system.v1.recurrent_route_count()

                diag_lines = [
                    f"V1 Cortex: {v1_res.active_count}/{system.v1.neurons} SDR | Novelty: {novelty_pct:4.1f}% | Settling: {v1_res.feedback_steps} cycles | Lateral Routes: {routes:,}",
                    f"Dual Routing: Ventral-Dominant: {assoc_res.strongly_ventral_count} | Dorsal-Dominant: {assoc_res.strongly_dorsal_count} | Shared: {assoc_res.shared_count}",
                    f"Ventral Stream: {assoc_res.ventral.active_count} -> Temporal Form Stability: {temp_res.stability*100:.1f}% (Invariance Index)",
                    f"Dorsal Stream: {assoc_res.dorsal.active_count} -> Action State: [{pari_res.action_label}] | Kinetic Energy: {pari_res.motion_energy*1000.0:.2f} mU",
                ]

                for i, line_txt in enumerate(diag_lines):
                    col = (0, 240, 255) if i == 0 else (220, 220, 220)
                    cv2.putText(dash_tile, line_txt, (18, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)

                # Novelty Meter Bar
                cv2.rectangle(dash_tile, (18, 132), (dash_w - 25, 148), (35, 35, 45), -1)
                bar_w = int((novelty_pct / 100.0) * (dash_w - 43))
                bar_col = (40, 90, 255) if novelty_pct > 25 else (0, 230, 180)
                cv2.rectangle(dash_tile, (18, 132), (18 + max(0, min(dash_w - 43, bar_w)), 148), bar_col, -1)
                cv2.putText(dash_tile, f"NOVELTY / SENSORY SURPRISE: {novelty_pct:.1f}%", (24, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(dash_tile, "Hardware Released. Press [k] or Click Button to reconnect.", (25, r3_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (160, 160, 160), 1, cv2.LINE_AA)

            row3 = np.hstack([route_tile, dash_tile])

            final_cockpit = np.vstack([header, row1, row2, row3])

            cv2.imshow(window, final_cockpit)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("k"), ord(" "), ord("p")):
                toggle_camera()
            elif key == ord("l"):
                learning_on = not learning_on
                print(f"LEARNING: {'[ON]' if learning_on else '[FROZEN]'}")
            elif key == ord("]"):
                system.v1.learning_rate = min(0.20, system.v1.learning_rate + 0.005)
                print(f"V1 Learning Rate increased: {system.v1.learning_rate:.3f}")
            elif key == ord("["):
                system.v1.learning_rate = max(0.002, system.v1.learning_rate - 0.005)
                print(f"V1 Learning Rate decreased: {system.v1.learning_rate:.3f}")
            elif key == ord("r"):
                system.reset_activity()
                print("All cortical recurrent states reset.")
            elif key == ord("e"):
                system.reset_expectation()
                print("Baseline sensory expectation reset.")

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(">> Camera released and windows closed.")
