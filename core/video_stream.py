# =============================================================================
# core/video_stream.py — Unified video source abstraction
# =============================================================================

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto


class SourceType(Enum):
    WEBCAM   = auto()
    IP_CAM   = auto()
    VIDEO_FILE = auto()


@dataclass
class StreamConfig:
    source_type: SourceType = SourceType.WEBCAM
    device_id:   int        = 0           # for WEBCAM
    url:         str        = ""          # for IP_CAM
    file_path:   str        = ""          # for VIDEO_FILE
    width:       int        = 1280
    height:      int        = 720
    fps_limit:   int        = 30


class VideoStream:
    """
    Context-manager compatible video stream.

    Usage::

        cfg = StreamConfig(source_type=SourceType.WEBCAM, device_id=0)
        with VideoStream(cfg) as stream:
            for frame in stream:
                ...  # frame is a BGR numpy array
    """

    def __init__(self, config: StreamConfig):
        self.config = config
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    def _open(self) -> cv2.VideoCapture:
        cfg = self.config
        if cfg.source_type == SourceType.WEBCAM:
            src = cfg.device_id
        elif cfg.source_type == SourceType.IP_CAM:
            src = cfg.url
        else:  # VIDEO_FILE
            src = cfg.file_path

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source: {src!r}. "
                "Check camera index / URL / file path."
            )

        if cfg.source_type == SourceType.WEBCAM:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            cap.set(cv2.CAP_PROP_FPS,          cfg.fps_limit)

        return cap

    # ------------------------------------------------------------------
    def open(self) -> "VideoStream":
        self._cap = self._open()
        return self

    def release(self):
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None

    # ------------------------------------------------------------------
    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        return ret, frame

    @property
    def total_frames(self) -> int:
        """Return total frame count for video files (-1 for live streams)."""
        if self._cap and self.config.source_type == SourceType.VIDEO_FILE:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1

    @property
    def native_fps(self) -> float:
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return 30.0

    # ------------------------------------------------------------------
    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.release()

    def __iter__(self):
        while self._cap and self._cap.isOpened():
            ret, frame = self.read_frame()
            if not ret:
                break
            yield frame


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def list_webcams(max_test: int = 5) -> list[int]:
    """Return indices of available webcam devices."""
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available
