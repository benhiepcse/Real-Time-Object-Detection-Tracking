"""OpenCV drawing helpers for tracked objects."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from src.tracker import Track


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Return a stable, bright BGR color for a track ID."""
    return (
        64 + (track_id * 37) % 192,
        64 + (track_id * 17) % 192,
        64 + (track_id * 29) % 192,
    )


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable[Track],
    *,
    fps: float | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Draw boxes, class labels, track IDs and optional processing FPS."""
    canvas = frame.copy() if copy else frame
    height, width = canvas.shape[:2]

    for track in tracks:
        x1, y1, x2, y2 = (int(round(value)) for value in track.xyxy)
        x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
        y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
        color = color_for_id(track.track_id)
        label = f"{track.class_name} #{track.track_id} {track.confidence:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(canvas, (x1, label_top), (min(width - 1, x1 + text_width + 6), y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, max(text_height + 1, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    if fps is not None:
        cv2.putText(
            canvas,
            f"Processing: {fps:.1f} FPS",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 255, 40),
            2,
            cv2.LINE_AA,
        )
    return canvas
