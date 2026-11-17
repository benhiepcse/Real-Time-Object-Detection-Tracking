"""YOLO object detector with a small, testable public interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """One detection in ``xyxy`` pixel coordinates."""

    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class YOLODetector:
    """Load an Ultralytics YOLO model and perform frame-level inference.

    The Ultralytics import is intentionally delayed until construction.  This
    keeps utility scripts importable even on machines where the ML dependencies
    have not been installed yet.
    """

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        *,
        device: str | int | None = None,
        confidence: float = 0.25,
        iou: float = 0.7,
        image_size: int = 640,
        classes: Iterable[int] | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.model = YOLO(weights)
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.classes = list(classes) if classes is not None else None

    def predict_raw(self, frame: np.ndarray, *, verbose: bool = False) -> Any:
        """Return the first native Ultralytics result for ``frame``."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            classes=self.classes,
            device=self.device,
            verbose=verbose,
        )
        if not results:
            raise RuntimeError("YOLO returned no result for the frame")
        return results[0]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects and convert tensor results to plain Python values."""
        result = self.predict_raw(frame)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names
        detections: list[Detection] = []
        for xyxy, confidence, class_id in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
        ):
            class_index = int(class_id)
            name = names.get(class_index, str(class_index)) if isinstance(names, dict) else names[class_index]
            detections.append(
                Detection(
                    xyxy=tuple(float(value) for value in xyxy),
                    confidence=float(confidence),
                    class_id=class_index,
                    class_name=str(name),
                )
            )
        return detections
