"""Live diagnostic window for the early retinal signal branches.

The UI does not define neural behavior. It shows the camera, measured brightness,
positive/negative temporal variation, local spatial contrast, and the first
one-direction temporal-spatial flow field owned by neural_field.py.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from neural_field import RetinalSignalPathway
from vision_input import VisualInputExtractor


class VisualPlayUI:
    WINDOW = "Visual Play - Retinal Signals"
    WIDTH = 1440
    HEIGHT = 900
    FIELD_COLS = 256
    FIELD_ROWS = 144

    def __init__(self) -> None:
        self.extractor = VisualInputExtractor(
            cols=self.FIELD_COLS,
            rows=self.FIELD_ROWS,
            mirror=True,
        )
        self.pathway = RetinalSignalPathway(
            rows=self.FIELD_ROWS,
            cols=self.FIELD_COLS,
        )

        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_on = False
        self.paused = False
        self.last_frame: Optional[np.ndarray] = None
        self.last_brightness: Optional[np.ndarray] = None
        self.last_on_activity: Optional[np.ndarray] = None
        self.last_off_activity: Optional[np.ndarray] = None
        self.last_contrast_activity: Optional[np.ndarray] = None
        self.last_rightward_activity: Optional[np.ndarray] = None

        self.fps = 0.0
        self._last_time = time.monotonic()
        self.status = "k camera | p pause | r reset retinal state | q quit"

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        candidates = []
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            candidates.append((0, cv2.CAP_AVFOUNDATION))
        candidates.append((0, cv2.CAP_ANY))

        for index, backend in candidates:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            cap.release()
        return None

    def toggle_camera(self) -> None:
        if self.camera_on:
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self.camera_on = False
            self.status = "Camera disconnected."
            return

        self.cap = self._open_camera()
        self.camera_on = bool(self.cap is not None and self.cap.isOpened())
        self.status = "Camera connected." if self.camera_on else "Camera unavailable."

    @staticmethod
    def _blank(height: int, width: int, value: int = 8) -> np.ndarray:
        return np.full((height, width, 3), value, dtype=np.uint8)

    @staticmethod
    def _map_to_bgr(values: np.ndarray) -> np.ndarray:
        clean = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
        gray = (clean * 255.0).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @classmethod
    def _fit(cls, image: np.ndarray, width: int, height: int, *, smooth: bool) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        src_h, src_w = image.shape[:2]
        if src_h < 1 or src_w < 1:
            return cls._blank(height, width, 4)

        scale = min(width / src_w, height / src_h)
        fit_w = max(1, int(round(src_w * scale)))
        fit_h = max(1, int(round(src_h * scale)))
        interpolation = cv2.INTER_AREA if smooth and scale < 1.0 else (
            cv2.INTER_LINEAR if smooth else cv2.INTER_NEAREST
        )
        resized = cv2.resize(image, (fit_w, fit_h), interpolation=interpolation)

        canvas = cls._blank(height, width, 4)
        x = (width - fit_w) // 2
        y = (height - fit_h) // 2
        canvas[y : y + fit_h, x : x + fit_w] = resized
        return canvas

    @staticmethod
    def _text(
        canvas: np.ndarray,
        text: str,
        x: int,
        y: int,
        scale: float = 0.48,
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            canvas,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (230, 230, 235),
            thickness,
            cv2.LINE_AA,
        )

    def _tile(
        self,
        image: np.ndarray,
        title: str,
        subtitle: str,
        width: int,
        height: int,
        *,
        smooth: bool = False,
    ) -> np.ndarray:
        tile = self._blank(height, width)
        header = 54
        tile[header:, :] = self._fit(image, width, height - header, smooth=smooth)
        cv2.rectangle(tile, (0, 0), (width - 1, header - 1), (20, 20, 24), -1)
        self._text(tile, title, 12, 23, 0.50)
        self._text(tile, subtitle, 12, 44, 0.32)
        cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (65, 65, 72), 1)
        return tile

    def _camera_image(self) -> np.ndarray:
        if self.last_frame is None or not self.camera_on:
            image = self._blank(220, 360, 4)
            self._text(image, "CAMERA OFF", 115, 120, 0.65, 2)
            return image
        return cv2.flip(self.last_frame, 1)

    def _map_or_blank(self, values: Optional[np.ndarray]) -> np.ndarray:
        if values is None:
            return self._blank(self.FIELD_ROWS, self.FIELD_COLS, 4)
        return self._map_to_bgr(values)

    def compose(self) -> np.ndarray:
        canvas = self._blank(self.HEIGHT, self.WIDTH, 5)

        header_h = 70
        self._text(canvas, "VISUAL PLAY - EARLY RETINAL SIGNALS", 18, 30, 0.66, 2)
        stats = (
            f"{self.pathway.sensory_location_count:,} sensory locations | "
            f"{self.pathway.neuron_count:,} graded branch neurons | {self.fps:4.1f} FPS"
        )
        self._text(canvas, stats, 18, 56, 0.40)
        self._text(canvas, self.status[:90], 900, 38, 0.36)

        gap = 12
        top_h = 350
        lower_h = (self.HEIGHT - header_h - top_h - gap * 4) // 2
        tile_w = (self.WIDTH - gap * 3) // 2

        brightness = self._map_or_blank(self.last_brightness)
        on_activity = self._map_or_blank(self.last_on_activity)
        off_activity = self._map_or_blank(self.last_off_activity)
        contrast_activity = self._map_or_blank(self.last_contrast_activity)
        rightward_activity = self._map_or_blank(self.last_rightward_activity)

        resolution = f"{self.FIELD_COLS} x {self.FIELD_ROWS}"
        branch_count = self.pathway.on_field.neuron_count

        rows = [
            [
                self._tile(
                    self._camera_image(),
                    "WEBCAM",
                    "physical input",
                    tile_w,
                    top_h,
                    smooth=True,
                ),
                self._tile(
                    brightness,
                    "BRIGHTNESS",
                    f"measured {resolution} luminance",
                    tile_w,
                    top_h,
                ),
            ],
            [
                self._tile(
                    on_activity,
                    "POSITIVE VARIATION",
                    f"{branch_count:,} neurons: brighter than adapting baseline",
                    tile_w,
                    lower_h,
                ),
                self._tile(
                    off_activity,
                    "NEGATIVE VARIATION",
                    f"{branch_count:,} neurons: darker than adapting baseline",
                    tile_w,
                    lower_h,
                ),
            ],
            [
                self._tile(
                    contrast_activity,
                    "LOCAL CONTRAST",
                    f"{branch_count:,} neurons: center vs immediate surround",
                    tile_w,
                    lower_h,
                ),
                self._tile(
                    rightward_activity,
                    "RIGHTWARD FLOW  ->",
                    f"{branch_count:,} neurons: left-before-right opponent timing",
                    tile_w,
                    lower_h,
                ),
            ],
        ]

        y_positions = [
            header_h + gap,
            header_h + gap * 2 + top_h,
            header_h + gap * 3 + top_h + lower_h,
        ]
        heights = [top_h, lower_h, lower_h]

        for row, y, height in zip(rows, y_positions, heights):
            for index, tile in enumerate(row):
                x = gap + index * (tile_w + gap)
                canvas[y : y + height, x : x + tile_w] = tile
        return canvas

    def reset(self) -> None:
        self.extractor.reset()
        self.pathway.reset()
        self.last_brightness = None
        self.last_on_activity = None
        self.last_off_activity = None
        self.last_contrast_activity = None
        self.last_rightward_activity = None
        self.status = "Retinal state reset; next frame will seed temporal baseline."

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.WIDTH, self.HEIGHT)
        self.toggle_camera()

        try:
            running = True
            while running:
                now = time.monotonic()
                dt = max(now - self._last_time, 1e-6)
                self._last_time = now
                self.fps = 0.90 * self.fps + 0.10 * (1.0 / dt)

                if self.camera_on and not self.paused and self.cap is not None:
                    ok, frame = self.cap.read()
                    if ok:
                        self.last_frame = frame.copy()
                        visual_input = self.extractor.extract(frame)
                        result = self.pathway.process(
                            visual_input.brightness,
                            timestamp_ms=now * 1000.0,
                        )
                        self.last_brightness = visual_input.brightness.copy()
                        self.last_on_activity = result.on_activity
                        self.last_off_activity = result.off_activity
                        self.last_contrast_activity = result.contrast_activity
                        self.last_rightward_activity = result.rightward_activity

                cv2.imshow(self.WINDOW, self.compose())
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    running = False
                elif key == ord("k"):
                    self.toggle_camera()
                elif key == ord("p"):
                    self.paused = not self.paused
                    self.status = "Paused." if self.paused else "Running."
                elif key == ord("r"):
                    self.reset()
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    VisualPlayUI().run()
