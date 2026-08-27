import time
import cv2
import numpy as np

# Resolution presets: (COLS, ROWS, Name)
PRESETS = [
    (80, 45, "Low (80x45)"),
    (160, 90, "Medium (160x90)"),
    (320, 180, "High (320x180)"),
    (480, 270, "Ultra HD (480x270)"),
    (960, 540, "Cinema HD (960x540)"),
]

COLORMAPS = [
    ("Plasma (Cyberpunk)", cv2.COLORMAP_PLASMA),
    ("Inferno", cv2.COLORMAP_INFERNO),
    ("Viridis", cv2.COLORMAP_VIRIDIS),
    ("Turbo Spectrum", cv2.COLORMAP_TURBO),
    ("Ocean Bone", cv2.COLORMAP_BONE),
    ("Monochrome Noir", None),
    ("Twilight", cv2.COLORMAP_TWILIGHT_SHIFTED),
]

MODES = [
    ("Rich Tonal Hybrid", "Balanced texture + luminance with full tonal range"),
    ("CLAHE Equalized", "Maximum textural gradient depth across every zone"),
    ("Gradient Flow", "Edge vector magnitude & high-frequency texture"),
    ("Pure Contrast (StdDev)", "Adaptive local statistical variance"),
]

# State & Tuning Parameters
preset_idx = 3  # Ultra HD (480x270)
COLS, ROWS = PRESETS[preset_idx][0], PRESETS[preset_idx][1]
colormap_idx = 0  # Default: Plasma
mode_idx = 0      # Rich Tonal Hybrid default
gamma = 0.60      # Midtone curve
gain = 1.15       # Contrast dynamic gain
mirror = True     # Natural mirror orientation
invert = False    # Invert black/white polarity
smooth = True     # Temporal EMA noise stabilization
smooth_alpha = 0.65
show_osd = True
is_fullscreen = False
camera_on = True  # Physical camera connection state


def open_camera():
    """Physically open the macOS webcam device."""
    c = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not c.isOpened():
        c = cv2.VideoCapture(0)
    if c.isOpened():
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return c


def toggle_camera():
    """Physically connect or disconnect the camera hardware."""
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


# Initial hardware connection
cap = open_camera()
if cap is None or not cap.isOpened():
    print("Error: Could not access webcam on startup.")
    exit(1)

window = "Visual Play - Contrast Engine"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 1280, 780)

clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))

buttons = {}


def on_mouse(event, x, y, flags, param):
    global colormap_idx, mode_idx, mirror, preset_idx, COLS, ROWS, prev_signals
    if event == cv2.EVENT_LBUTTONDOWN:
        for btn_name, (x1, y1, x2, y2) in buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if btn_name == "power":
                    toggle_camera()
                elif btn_name == "theme":
                    colormap_idx = (colormap_idx + 1) % len(COLORMAPS)
                elif btn_name == "mode":
                    mode_idx = (mode_idx + 1) % len(MODES)
                elif btn_name == "mirror":
                    mirror = not mirror
                elif btn_name == "res_up":
                    preset_idx = min(len(PRESETS) - 1, preset_idx + 1)
                    COLS, ROWS = PRESETS[preset_idx][0], PRESETS[preset_idx][1]
                    prev_signals = None
                elif btn_name == "res_down":
                    preset_idx = max(0, preset_idx - 1)
                    COLS, ROWS = PRESETS[preset_idx][0], PRESETS[preset_idx][1]
                    prev_signals = None


cv2.setMouseCallback(window, on_mouse)

prev_signals = None
prev_time = time.time()
fps = 0.0
last_display = None

print("=" * 70)
print("       VISUAL PLAY - CONTRAST & TONE ENGINE")
print("=" * 70)
print("Hardware Killswitch & Controls:")
print("  [k], [SPACE], [p] : Physically Connect / Disconnect Camera (Killswitch)")
print("  [1] - [5]         : Resolution Presets")
print("  [m]               : Cycle Contrast Modes")
print("  [c]               : Cycle Color Themes")
print("  [g] / [h]         : Adjust Gamma (Midtones)")
print("  [ [ ] / [ ] ]     : Adjust Contrast Gain")
print("  [i]               : Invert Polarity")
print("  [x]               : Toggle Mirror Flip")
print("  [t]               : Toggle Noise Smoothing")
print("  [o]               : Toggle HUD")
print("  [f]               : Toggle Fullscreen")
print("  [q]               : Quit")
print("=" * 70)

try:
    while True:
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        target_w, target_h = 1280, 720
        header_h = 48

        if camera_on and cap is not None:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            cell_h = max(1, h // ROWS)
            cell_w = max(1, w // COLS)

            crop_h = ROWS * cell_h
            crop_w = COLS * cell_w
            cropped = gray[:crop_h, :crop_w]

            cells = cropped.reshape(ROWS, cell_h, COLS, cell_w)

            mode_name = MODES[mode_idx][0]

            if mode_name == "Rich Tonal Hybrid":
                mean_c = np.mean(cells, axis=(1, 3)).astype(np.float32)
                std_c = np.std(cells, axis=(1, 3)).astype(np.float32)
                signals = 0.65 * std_c + 0.35 * (mean_c * (35.0 / 255.0))
            elif mode_name == "CLAHE Equalized":
                signals = np.std(cells, axis=(1, 3)).astype(np.float32)
            elif mode_name == "Gradient Flow":
                mean_c = np.mean(cells, axis=(1, 3)).astype(np.float32)
                gx = cv2.Sobel(mean_c, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(mean_c, cv2.CV_32F, 0, 1, ksize=3)
                grad = cv2.magnitude(gx, gy)
                signals = 0.55 * np.std(cells, axis=(1, 3)) + 0.45 * grad
            else:
                signals = np.std(cells, axis=(1, 3)).astype(np.float32)

            if smooth:
                if prev_signals is not None and prev_signals.shape == signals.shape:
                    signals = smooth_alpha * signals + (1.0 - smooth_alpha) * prev_signals
                prev_signals = signals.copy()
            else:
                prev_signals = None

            p_low = np.percentile(signals, 1.5)
            p_high = np.percentile(signals, 98.5)
            dyn_range = max(p_high - p_low, 1e-4)

            norm = np.clip(((signals - p_low) / dyn_range) * gain, 0.0, 1.0)
            norm = np.power(norm, gamma)

            if invert:
                norm = 1.0 - norm

            uint8_grid = (norm * 255).astype(np.uint8)

            if mode_name in ("CLAHE Equalized", "Rich Tonal Hybrid"):
                uint8_grid = clahe.apply(uint8_grid)

            display = cv2.resize(uint8_grid, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            theme_name, cm_code = COLORMAPS[colormap_idx]
            if cm_code is not None:
                display = cv2.applyColorMap(display, cm_code)
            else:
                display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

            last_display = display.copy()

        else:
            # Standby screen when physically disconnected
            display = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            cv2.rectangle(display, (target_w // 2 - 300, target_h // 2 - 50), (target_w // 2 + 300, target_h // 2 + 50), (20, 20, 30), -1)
            cv2.rectangle(display, (target_w // 2 - 300, target_h // 2 - 50), (target_w // 2 + 300, target_h // 2 + 50), (50, 50, 220), 2)
            cv2.putText(display, "CAMERA PHYSICALLY DISCONNECTED", (target_w // 2 - 250, target_h // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "Hardware released (Green light OFF) | Press [k] or click button to reconnect", (target_w // 2 - 280, target_h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
            time.sleep(0.03)

        # --- TOP CONTROL BAR ---
        header = np.zeros((header_h, target_w, 3), dtype=np.uint8)

        # 1. Camera Hardware Killswitch Button
        buttons["power"] = (12, 6, 210, 42)
        if camera_on:
            cv2.rectangle(header, (12, 6), (210, 42), (30, 180, 50), -1)
            cv2.rectangle(header, (12, 6), (210, 42), (80, 255, 120), 2)
            cv2.putText(header, "[CAM: ON - DISCONNECT (k)]", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(header, (12, 6), (210, 42), (40, 40, 200), -1)
            cv2.rectangle(header, (12, 6), (210, 42), (80, 80, 255), 2)
            cv2.putText(header, "[CAM: OFF - CONNECT (k)]", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Theme Button
        buttons["theme"] = (220, 6, 370, 42)
        cv2.rectangle(header, (220, 6), (370, 42), (50, 40, 70), -1)
        cv2.rectangle(header, (220, 6), (370, 42), (160, 100, 220), 1)
        cv2.putText(header, f"THEME: {COLORMAPS[colormap_idx][0][:9]}", (228, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 200, 255), 1, cv2.LINE_AA)

        # 3. Mode Button
        buttons["mode"] = (380, 6, 540, 42)
        cv2.rectangle(header, (380, 6), (540, 42), (40, 50, 60), -1)
        cv2.rectangle(header, (380, 6), (540, 42), (100, 160, 200), 1)
        cv2.putText(header, f"MODE: {MODES[mode_idx][0][:11]}", (388, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 230, 255), 1, cv2.LINE_AA)

        # 4. Mirror Button
        buttons["mirror"] = (550, 6, 660, 42)
        cv2.rectangle(header, (550, 6), (660, 42), (40, 40, 50), -1)
        cv2.rectangle(header, (550, 6), (660, 42), (120, 120, 140), 1)
        cv2.putText(header, f"MIRROR: {'ON' if mirror else 'OFF'}", (558, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

        # 5. Resolution +/- Buttons
        buttons["res_down"] = (670, 6, 725, 42)
        cv2.rectangle(header, (670, 6), (725, 42), (35, 35, 45), -1)
        cv2.rectangle(header, (670, 6), (725, 42), (100, 100, 120), 1)
        cv2.putText(header, "RES -", (678, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        buttons["res_up"] = (735, 6, 790, 42)
        cv2.rectangle(header, (735, 6), (790, 42), (35, 35, 45), -1)
        cv2.rectangle(header, (735, 6), (790, 42), (100, 100, 120), 1)
        cv2.putText(header, "RES +", (743, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Right info
        res_text = f"Grid: {COLS}x{ROWS} | {fps:.1f} FPS"
        cv2.putText(header, res_text, (target_w - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)

        # Optional In-Game HUD overlay
        if show_osd and camera_on:
            total_cells = COLS * ROWS
            overlay = display.copy()
            cv2.rectangle(overlay, (12, 12), (1020, 75), (10, 10, 15), -1)
            cv2.addWeighted(overlay, 0.70, display, 0.30, 0, display)

            line1 = f"Mode: {MODES[mode_idx][0]}  |  Theme: {COLORMAPS[colormap_idx][0]}  |  Grid: {COLS}x{ROWS} ({total_cells:,} px)"
            line2 = f"Killswitch: [k]  Gamma: {gamma:.2f} [g/h]  Gain: {gain:.2f} [[/]]  Smooth: {'ON' if smooth else 'OFF'} [t]  HUD: [o]"

            cv2.putText(display, line1, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 255), 1, cv2.LINE_AA)
            cv2.putText(display, line2, (22, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 215, 215), 1, cv2.LINE_AA)

        final_layout = np.vstack([header, display])
        cv2.imshow(window, final_layout)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("k"), ord(" "), ord("p")):
            toggle_camera()
        elif ord("1") <= key <= ord("5"):
            preset_idx = key - ord("1")
            COLS, ROWS = PRESETS[preset_idx][0], PRESETS[preset_idx][1]
            prev_signals = None
        elif key in (ord("+"), ord("=")):
            ROWS = min(ROWS + 27, h // 2)
            COLS = min(COLS + 48, w // 2)
            prev_signals = None
        elif key in (ord("-"), ord("_")):
            ROWS = max(18, ROWS - 27)
            COLS = max(32, COLS - 48)
            prev_signals = None
        elif key == ord("m"):
            mode_idx = (mode_idx + 1) % len(MODES)
        elif key == ord("c"):
            colormap_idx = (colormap_idx + 1) % len(COLORMAPS)
        elif key == ord("g"):
            gamma = max(0.15, gamma - 0.05)
        elif key == ord("h"):
            gamma = min(2.5, gamma + 0.05)
        elif key == ord("["):
            gain = max(0.2, gain - 0.1)
        elif key == ord("]"):
            gain = min(3.5, gain + 0.1)
        elif key == ord("x"):
            mirror = not mirror
        elif key == ord("i"):
            invert = not invert
        elif key == ord("t"):
            smooth = not smooth
        elif key == ord("o"):
            show_osd = not show_osd
        elif key == ord("f"):
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, prop)

finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(">> Webcam and OpenCV windows cleanly closed.")