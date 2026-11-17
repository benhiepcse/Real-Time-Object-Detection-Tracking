"""Stateful ByteTrack integration built on Ultralytics tracking mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class Track:
    """A tracked object returned in ordinary Python types."""

    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class ByteTracker:
    """Maintain ByteTrack identity state across consecutive frames."""

    def __init__(
        self,
        model: Any,
        *,
        tracker_config: str = "bytetrack.yaml",
        device: str | int | None = None,
        confidence: float = 0.25,
        iou: float = 0.7,
        image_size: int = 640,
        classes: Iterable[int] | None = None,
    ) -> None:
        if not hasattr(model, "track"):
            raise TypeError("model must be an Ultralytics YOLO model")
        if tracker_config != "bytetrack.yaml" and not Path(tracker_config).is_file():
            raise FileNotFoundError(f"Tracker config not found: {tracker_config}")

        self.model = model
        self.tracker_config = tracker_config
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.classes = list(classes) if classes is not None else None

    def reset(self) -> None:
        """Discard tracker callbacks/state before starting another stream."""
        predictors = getattr(self.model, "predictor", None)
        if predictors is not None:
            for tracker in getattr(predictors, "trackers", []) or []:
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    reset()

    def update_raw(self, frame: np.ndarray, *, verbose: bool = False) -> Any:
        """Track one frame and return the native Ultralytics result."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            classes=self.classes,
            device=self.device,
            verbose=verbose,
        )
        if not results:
            raise RuntimeError("YOLO tracker returned no result for the frame")
        return results[0]

    def update(self, frame: np.ndarray) -> list[Track]:
        """Track a frame and return only confirmed tracks (boxes with IDs)."""
        result = self.update_raw(frame)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []

        names = result.names
        tracks: list[Track] = []
        for xyxy, confidence, class_id, track_id in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
            boxes.id.int().cpu().tolist(),
        ):
            class_index = int(class_id)
            name = names.get(class_index, str(class_index)) if isinstance(names, dict) else names[class_index]
            tracks.append(
                Track(
                    track_id=int(track_id),
                    xyxy=tuple(float(value) for value in xyxy),
                    confidence=float(confidence),
                    class_id=class_index,
                    class_name=str(name),
                )
            )
        return tracks
