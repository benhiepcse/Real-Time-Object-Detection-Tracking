import math
from contextlib import contextmanager
from pathlib import Path

import cv2


def parse_source(value: str) -> str | int:
    if value.isascii() and value.isdigit():
        return int(value)

    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    return str(path)


@contextmanager
def open_capture(source: str | int):
    capture = cv2.VideoCapture(source)

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        yield capture
    finally:
        capture.release()


def get_output_fps(capture) -> float:
    fps = capture.get(cv2.CAP_PROP_FPS)

    if math.isfinite(fps) and fps > 0:
        return fps

    return 30.0


@contextmanager
def open_writer(path: Path, fps: float, frame_size: tuple[int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)

    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), codec, fps, frame_size)

    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {path}")

        yield writer
    finally:
        writer.release()
