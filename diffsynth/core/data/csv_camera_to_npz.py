#!/usr/bin/env python3
import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConvertResult:
    sample_id: str
    path: str
    num_frames: int
    sample_dir: Path


def _parse_list(value: str):
    value = value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def parse_c2w(value: str) -> np.ndarray:
    c2w = np.asarray(_parse_list(value), dtype=np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"c2w must have shape (N, 3, 4) or (N, 4, 4), got {c2w.shape}")

    if c2w.shape[1:] == (4, 4):
        return c2w

    full = np.tile(np.eye(4, dtype=np.float32)[None], (c2w.shape[0], 1, 1))
    full[:, :3, :4] = c2w
    return full


def parse_intrinsics(
    value: str, num_frames: int, intrinsic_order: str = "fx_fy_cx_cy"
) -> np.ndarray:
    intrinsics = np.asarray(_parse_list(value), dtype=np.float32)
    if intrinsics.shape == (4,):
        if intrinsic_order == "cx_cy_fx_fy":
            intrinsics = intrinsics[[2, 3, 0, 1]]
        return np.repeat(intrinsics[None, :], repeats=num_frames, axis=0)

    if intrinsics.shape == (num_frames, 4):
        if intrinsic_order == "cx_cy_fx_fy":
            intrinsics = intrinsics[:, [2, 3, 0, 1]]
        return intrinsics.astype(np.float32, copy=False)

    if intrinsics.shape == (3, 3):
        flat = np.array(
            [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]],
            dtype=np.float32,
        )
        return np.repeat(flat[None, :], repeats=num_frames, axis=0)

    if intrinsics.shape == (num_frames, 3, 3):
        return np.stack(
            [
                intrinsics[:, 0, 0],
                intrinsics[:, 1, 1],
                intrinsics[:, 0, 2],
                intrinsics[:, 1, 2],
            ],
            axis=-1,
        ).astype(np.float32)

    raise ValueError(
        "intrinsic must be shape (4,), (N, 4), (3, 3), or (N, 3, 3), "
        f"got {intrinsics.shape}"
    )


def make_sample_id(video_path: str, id_mode: str = "stem_hash") -> str:
    path = Path(video_path)
    digest = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:8]

    if id_mode == "hash":
        return digest
    if id_mode == "stem":
        return path.stem
    if id_mode == "stem_hash":
        return f"{path.stem}__{digest}"
    if id_mode == "path":
        parts = [part for part in path.parts if part not in (path.anchor, os.sep)]
        safe = "__".join(parts)
        return safe.replace("/", "__").replace("\\", "__")

    raise ValueError(f"Unsupported id_mode: {id_mode}")


def iter_csv_rows(csv_path: Path, limit: int | None = None):
    csv.field_size_limit(sys.maxsize)
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"path", "c2w", "intrinsic"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            yield row


def write_camera_txt(path: Path, num_frames: int):
    with path.open("w", encoding="utf-8") as f:
        for i in range(num_frames):
            f.write(f"{i}: PINHOLE\n")


def maybe_link_rgb(video_path: str, sample_dir: Path):
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"Video path does not exist: {src}")

    rgb_dir = sample_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    dst = rgb_dir / "video.mp4"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def maybe_copy_rgb(video_path: str, sample_dir: Path):
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"Video path does not exist: {src}")

    rgb_dir = sample_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    dst = rgb_dir / "video.mp4"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def maybe_write_rgb(video_path: str, sample_dir: Path, link_rgb: bool, copy_rgb: bool):
    if link_rgb and copy_rgb:
        raise ValueError("link_rgb and copy_rgb are mutually exclusive")
    if link_rgb:
        maybe_link_rgb(video_path, sample_dir)
    if copy_rgb:
        maybe_copy_rgb(video_path, sample_dir)


def convert_row(
    row: dict[str, str],
    output_dir: Path,
    id_mode: str = "stem_hash",
    intrinsic_order: str = "fx_fy_cx_cy",
    overwrite: bool = False,
    link_rgb: bool = False,
    copy_rgb: bool = False,
) -> ConvertResult:
    video_path = row["path"]
    sample_id = make_sample_id(video_path, id_mode=id_mode)
    sample_dir = Path(output_dir) / sample_id
    pose_path = sample_dir / "pose" / "video.npz"
    intrinsic_path = sample_dir / "intrinsics" / "video.npz"

    if not overwrite and pose_path.exists() and intrinsic_path.exists():
        maybe_write_rgb(video_path, sample_dir, link_rgb=link_rgb, copy_rgb=copy_rgb)
        with np.load(pose_path) as f:
            num_frames = int(f["data"].shape[0])
        return ConvertResult(sample_id, video_path, num_frames, sample_dir)

    c2w = parse_c2w(row["c2w"])
    num_frames = int(c2w.shape[0])
    intrinsics = parse_intrinsics(
        row["intrinsic"], num_frames=num_frames, intrinsic_order=intrinsic_order
    )

    pose_path.parent.mkdir(parents=True, exist_ok=True)
    intrinsic_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(pose_path, data=c2w.astype(np.float32, copy=False))
    np.savez(intrinsic_path, data=intrinsics.astype(np.float32, copy=False))
    write_camera_txt(intrinsic_path.parent / "video_camera.txt", num_frames)

    maybe_write_rgb(video_path, sample_dir, link_rgb=link_rgb, copy_rgb=copy_rgb)

    return ConvertResult(sample_id, video_path, num_frames, sample_dir)


def convert_csv(
    csv_path: Path,
    output_dir: Path,
    id_mode: str = "stem_hash",
    intrinsic_order: str = "fx_fy_cx_cy",
    limit: int | None = None,
    overwrite: bool = False,
    link_rgb: bool = False,
    copy_rgb: bool = False,
    progress_every: int = 100,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "camera_manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["sample_id", "path", "num_frames", "sample_dir"],
        )
        writer.writeheader()

        for index, row in enumerate(iter_csv_rows(csv_path, limit=limit), start=1):
            try:
                result = convert_row(
                    row,
                    output_dir,
                    id_mode=id_mode,
                    intrinsic_order=intrinsic_order,
                    overwrite=overwrite,
                    link_rgb=link_rgb,
                    copy_rgb=copy_rgb,
                )
            except Exception as exc:
                print(f"[ERROR] row={index} path={row.get('path', '')}: {exc}")
                continue

            writer.writerow(
                {
                    "sample_id": result.sample_id,
                    "path": result.path,
                    "num_frames": result.num_frames,
                    "sample_dir": str(result.sample_dir),
                }
            )
            if progress_every > 0 and index % progress_every == 0:
                print(f"[{index}] last={result.sample_id} frames={result.num_frames}")

    print(f"Done. Manifest saved to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stream a large Unreal camera CSV into VIPE-style camera npz files."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=(
            "runjia.qian/train_data_unrealV4_third_person/"
            "Unreal_V4.1_path_c2w_intrinsic_action_camera_captioned.csv"
        ),
        help="Input CSV with path,c2w,intrinsic columns.",
    )
    parser.add_argument(
        "output_dir",
        help="Output root. Each row writes <output_dir>/<sample_id>/pose and intrinsics.",
    )
    parser.add_argument(
        "--id-mode",
        default="stem_hash",
        choices=["stem_hash", "stem", "hash", "path"],
        help="How to name each sample directory.",
    )
    parser.add_argument(
        "--intrinsic-order",
        default="fx_fy_cx_cy",
        choices=["fx_fy_cx_cy", "cx_cy_fx_fy"],
        help="Order used by the CSV intrinsic field before writing fx,fy,cx,cy.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process N rows.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pose/intrinsics npz files.",
    )
    rgb_group = parser.add_mutually_exclusive_group()
    rgb_group.add_argument(
        "--link-rgb",
        action="store_true",
        help="Also create rgb/video.mp4 as a symlink to the CSV path.",
    )
    rgb_group.add_argument(
        "--copy-rgb",
        action="store_true",
        help="Also copy the CSV video path to rgb/video.mp4.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print one progress line every N processed rows. Use 0 to disable.",
    )
    args = parser.parse_args()

    convert_csv(
        csv_path=Path(args.csv_path),
        output_dir=Path(args.output_dir),
        id_mode=args.id_mode,
        intrinsic_order=args.intrinsic_order,
        limit=args.limit,
        overwrite=args.overwrite,
        link_rgb=args.link_rgb,
        copy_rgb=args.copy_rgb,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
