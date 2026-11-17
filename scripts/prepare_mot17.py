"""Discover and validate MOT17 sequences before evaluation."""

from __future__ import annotations

import argparse
import configparser
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MOTSequence:
    path: Path
    name: str
    length: int
    width: int
    height: int
    fps: float
    has_ground_truth: bool


def find_dataset_root(path: Path) -> Path:
    """Accept either MOT17 root or a parent containing ``MOT17``."""
    candidates = (path, path / "MOT17")
    for candidate in candidates:
        if (candidate / "train").is_dir() or (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find MOT17 train/test below: {path}")


def inspect_sequence(path: Path, require_gt: bool = False) -> MOTSequence:
    info_path = path / "seqinfo.ini"
    image_dir = path / "img1"
    gt_path = path / "gt" / "gt.txt"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing {image_dir}")
    if require_gt and not gt_path.is_file():
        raise FileNotFoundError(f"Missing {gt_path}")

    parser = configparser.ConfigParser()
    parser.read(info_path, encoding="utf-8")
    section = parser["Sequence"]
    sequence = MOTSequence(
        path=path,
        name=section.get("name", path.name),
        length=section.getint("seqLength"),
        width=section.getint("imWidth"),
        height=section.getint("imHeight"),
        fps=section.getfloat("frameRate"),
        has_ground_truth=gt_path.is_file(),
    )
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    if len(images) != sequence.length:
        raise ValueError(
            f"{sequence.name}: seqLength={sequence.length}, but found {len(images)} images"
        )
    return sequence


def discover_sequences(root: Path, split: str, require_gt: bool) -> list[MOTSequence]:
    split_dir = find_dataset_root(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"MOT17 split not found: {split_dir}")
    return [inspect_sequence(path, require_gt) for path in sorted(split_dir.iterdir()) if path.is_dir()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an extracted MOT17 dataset")
    parser.add_argument("--root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--require-gt", action="store_true")
    args = parser.parse_args()
    try:
        sequences = discover_sequences(args.root, args.split, args.require_gt)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1
    for sequence in sequences:
        gt = "yes" if sequence.has_ground_truth else "no"
        print(f"{sequence.name:16} {sequence.length:5} frames  {sequence.width}x{sequence.height}  {sequence.fps:g} FPS  GT={gt}")
    print(f"Validated {len(sequences)} sequences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
