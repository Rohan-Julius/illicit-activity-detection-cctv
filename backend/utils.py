from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def mean_brightness_bgr(frame_bgr: np.ndarray) -> float:
    # Use HSV V channel for robust brightness
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return float(np.mean(v) / 255.0)


def parse_opencv_source(video_url: Optional[str]):
    if video_url is None:
        return 0
    s = str(video_url).strip()
    if s.isdigit():
        return int(s)
    return s


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return buf.tobytes()

