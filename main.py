"""Command-line entry point for real-time YOLO + ByteTrack inference."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import yaml

from src.detector import YOLODetector
from src.tracker import ByteTracker
from src.video_io import get_output_fps, open_capture, open_writer, parse_source
from src.visualize import draw_tracks


def load_config(path: Path) -> dict:
    """Read a YAML mapping, returning an empty mapping for an empty file."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("Config root must be a YAML mapping")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and track objects with YOLO + ByteTrack")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    parser.add_argument("--output", type=Path, help="Optional annotated MP4 output")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--weights", help="Override model weights from config")
    parser.add_argument("--device", help="cpu, 0, 0,1, etc.")
    parser.add_argument("--conf", type=float, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, help="Inference image size")
    parser.add_argument("--classes", type=int, nargs="*", help="Only these COCO class IDs")
    parser.add_argument("--show", action="store_true", help="Show a live preview; press q or Esc to stop")
    parser.add_argument("--max-frames", type=int, help="Stop after N frames (useful for testing)")
    return parser


def run(args: argparse.Namespace) -> dict[str, float | int]:
    config = load_config(args.config)
    model_cfg = config.get("model", {})
    tracker_cfg = config.get("tracker", {})

    detector = YOLODetector(
        weights=args.weights or model_cfg.get("weights", "yolo11n.pt"),
        device=args.device if args.device is not None else model_cfg.get("device"),
        confidence=args.conf if args.conf is not None else float(model_cfg.get("confidence", 0.25)),
        iou=args.iou if args.iou is not None else float(model_cfg.get("iou", 0.7)),
        image_size=args.imgsz or int(model_cfg.get("image_size", 640)),
        classes=args.classes if args.classes is not None else model_cfg.get("classes"),
    )
    tracker = ByteTracker(
        detector.model,
        tracker_config=str(tracker_cfg.get("config", "bytetrack.yaml")),
        device=detector.device,
        confidence=detector.confidence,
        iou=detector.iou,
        image_size=detector.image_size,
        classes=detector.classes,
    )

    source = parse_source(args.source)
    durations: deque[float] = deque(maxlen=30)
    processed = 0
    started = time.perf_counter()

    with open_capture(source) as capture:
        fps = get_output_fps(capture)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            ok, first_frame = capture.read()
            if not ok:
                raise RuntimeError("Video source opened but returned no frames")
            height, width = first_frame.shape[:2]
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        writer_context = open_writer(args.output, fps, (width, height)) if args.output else _null_writer()
        with writer_context as writer:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                tick = time.perf_counter()
                tracks = tracker.update(frame)
                durations.append(time.perf_counter() - tick)
                live_fps = len(durations) / sum(durations) if sum(durations) > 0 else 0.0
                annotated = draw_tracks(frame, tracks, fps=live_fps)

                if writer is not None:
                    writer.write(annotated)
                if args.show:
                    cv2.imshow("YOLO + ByteTrack", annotated)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        processed += 1
                        break

                processed += 1
                if args.max_frames is not None and processed >= args.max_frames:
                    break

    if args.show:
        cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    return {"frames": processed, "seconds": elapsed, "fps": processed / elapsed if elapsed else 0.0}


class _null_writer:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(
        f"Done: {summary['frames']} frames in {summary['seconds']:.2f}s "
        f"({summary['fps']:.2f} end-to-end FPS)"
    )
    if args.output:
        print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
