"""Benchmark YOLO + ByteTrack inference latency on a video source."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import YOLODetector
from src.tracker import ByteTracker
from src.video_io import open_capture, parse_source


def percentile(values: list[float], percentage: float) -> float:
    """Linear-interpolated percentile without requiring pandas/scipy."""
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty list")
    return float(np.percentile(np.asarray(values), percentage))


def benchmark(args: argparse.Namespace) -> dict[str, float | int]:
    detector = YOLODetector(
        args.weights,
        device=args.device,
        confidence=args.conf,
        iou=args.iou,
        image_size=args.imgsz,
        classes=args.classes,
    )
    tracker = ByteTracker(
        detector.model,
        device=detector.device,
        confidence=detector.confidence,
        iou=detector.iou,
        image_size=detector.image_size,
        classes=detector.classes,
    )
    samples: list[float] = []
    seen = 0
    with open_capture(parse_source(args.source)) as capture:
        while seen < args.frames:
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
                if not ok:
                    break
                tracker.reset()
            started = time.perf_counter()
            tracker.update(frame)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if seen >= args.warmup:
                samples.append(elapsed_ms)
            seen += 1

    if not samples:
        raise RuntimeError("No measured frames; reduce --warmup or check the source")
    total_seconds = sum(samples) / 1000.0
    return {
        "frames": len(samples),
        "fps": len(samples) / total_seconds,
        "mean_ms": statistics.fmean(samples),
        "p50_ms": percentile(samples, 50),
        "p95_ms": percentile(samples, 95),
        "p99_ms": percentile(samples, 99),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark model + tracker inference")
    parser.add_argument("--source", default="data/input/MOT17-02-FRCNN.mp4")
    parser.add_argument("--weights", default="weights/yolo11n.pt")
    parser.add_argument("--device")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--frames", type=int, default=220, help="Total frames including warm-up")
    args = parser.parse_args()
    if args.warmup < 0 or args.frames <= args.warmup:
        parser.error("--frames must be greater than --warmup >= 0")
    try:
        result = benchmark(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Measured frames: {result['frames']}")
    print(f"Throughput:      {result['fps']:.2f} FPS")
    print(f"Latency mean:    {result['mean_ms']:.2f} ms")
    print(f"Latency P50/P95/P99: {result['p50_ms']:.2f} / {result['p95_ms']:.2f} / {result['p99_ms']:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
