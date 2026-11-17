"""Run tracking on MOT17 image sequences and export MOTChallenge text files."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_mot17 import discover_sequences
from src.detector import YOLODetector
from src.tracker import ByteTracker, Track


def mot_line(frame_number: int, track: Track) -> str:
    """Serialize a track using MOTChallenge's 10-column result format."""
    x1, y1, x2, y2 = track.xyxy
    width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
    return (
        f"{frame_number},{track.track_id},{x1:.2f},{y1:.2f},"
        f"{width:.2f},{height:.2f},{track.confidence:.6f},-1,-1,-1\n"
    )


def evaluate_sequence(sequence, tracker: ByteTracker, output: Path, max_frames: int | None) -> tuple[int, float]:
    image_paths = sorted((sequence.path / "img1").glob("*.jpg"))
    if not image_paths:
        image_paths = sorted((sequence.path / "img1").glob("*.png"))
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    tracker.reset()
    started = time.perf_counter()
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for frame_number, image_path in enumerate(image_paths, start=1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Cannot read image: {image_path}")
            for track in tracker.update(frame):
                stream.write(mot_line(frame_number, track))
    elapsed = time.perf_counter() - started
    return len(image_paths), elapsed


def run_trackeval(trackeval_root: Path, dataset_root: Path, result_data_dir: Path, sequences: list[str]) -> None:
    script = trackeval_root / "scripts" / "run_mot_challenge.py"
    if not script.is_file():
        raise FileNotFoundError(f"TrackEval runner not found: {script}")
    command = [
        sys.executable, str(script),
        "--BENCHMARK", "MOT17",
        "--SPLIT_TO_EVAL", "train",
        "--GT_FOLDER", str(dataset_root),
        "--TRACKERS_FOLDER", str(result_data_dir.parent.parent),
        "--TRACKERS_TO_EVAL", result_data_dir.parent.name,
        "--SEQ_INFO", *sequences,
        "--METRICS", "HOTA", "CLEAR", "Identity",
    ]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO + ByteTrack predictions for MOT17")
    parser.add_argument("--root", type=Path, default=Path("data/mot17"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/mot17/YOLOByteTrack/data"),
        help="TrackEval-compatible tracker data directory",
    )
    parser.add_argument("--sequences", nargs="*", help="Sequence names; default is one FRCNN variant per scene")
    parser.add_argument("--weights", default="weights/yolo11n.pt")
    parser.add_argument("--device")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--trackeval", type=Path, help="Optional path to a TrackEval checkout")
    args = parser.parse_args()

    try:
        sequences = discover_sequences(args.root, "train", require_gt=True)
        requested = set(args.sequences or [])
        if requested:
            sequences = [sequence for sequence in sequences if sequence.name in requested]
            missing = requested - {sequence.name for sequence in sequences}
            if missing:
                raise ValueError(f"Unknown sequences: {', '.join(sorted(missing))}")
        else:
            sequences = [sequence for sequence in sequences if sequence.name.endswith("-FRCNN")]

        detector = YOLODetector(
            args.weights, device=args.device, confidence=args.conf,
            iou=args.iou, image_size=args.imgsz, classes=[0],
        )
        tracker = ByteTracker(
            detector.model, device=detector.device, confidence=detector.confidence,
            iou=detector.iou, image_size=detector.image_size, classes=[0],
        )
        for sequence in sequences:
            frames, elapsed = evaluate_sequence(
                sequence, tracker, args.output / f"{sequence.name}.txt", args.max_frames
            )
            print(f"{sequence.name}: {frames} frames, {frames / elapsed:.2f} FPS -> {args.output / (sequence.name + '.txt')}")
        if args.trackeval:
            dataset_root = sequences[0].path.parents[1] if sequences else args.root
            run_trackeval(args.trackeval, dataset_root, args.output, [s.name for s in sequences])
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
