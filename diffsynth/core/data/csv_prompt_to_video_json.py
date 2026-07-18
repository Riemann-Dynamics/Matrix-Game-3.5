#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from csv_camera_to_npz import make_sample_id
except ImportError:
    from diffsynth.core.data.csv_camera_to_npz import make_sample_id


DYNAMIC_SEPARATOR = "[DYNAMIC]"


@dataclass(frozen=True)
class PromptConvertResult:
    sample_id: str
    path: str
    num_clips: int
    num_frames: int
    json_path: Path


def split_prompt(text: str) -> tuple[str, str]:
    if DYNAMIC_SEPARATOR not in text:
        return text.strip(), ""

    start, dynamic = text.split(DYNAMIC_SEPARATOR, 1)
    return start.strip(), dynamic.strip()


def iter_prompt_rows(csv_path: Path, limit_rows: int | None = None):
    csv.field_size_limit(sys.maxsize)
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"path", "text", "start_frame", "end_frame"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for index, row in enumerate(reader):
            if limit_rows is not None and index >= limit_rows:
                break
            yield row


def iter_prompt_groups(csv_path: Path, limit_rows: int | None = None):
    current_path = None
    current_rows = []
    seen_paths = set()

    for row in iter_prompt_rows(csv_path, limit_rows=limit_rows):
        row_path = row["path"]
        if current_path is None:
            current_path = row_path
        if row_path != current_path:
            seen_paths.add(current_path)
            yield current_rows
            if row_path in seen_paths:
                raise ValueError(
                    "CSV rows are not grouped by path; refusing to merge non-contiguous "
                    f"clips for {row_path}"
                )
            current_path = row_path
            current_rows = []
        current_rows.append(row)

    if current_rows:
        yield current_rows


def _frame_range(row: dict[str, str]) -> range:
    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])
    if end_frame < start_frame:
        raise ValueError(
            f"end_frame must be >= start_frame, got {start_frame} -> {end_frame}"
        )
    return range(start_frame, end_frame)


def rows_to_expanded_detailed(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    detailed = {}
    for row in sorted(rows, key=lambda item: int(item["start_frame"])):
        start, dynamic = split_prompt(row["text"])
        entry = {"start": start, "dynamic": dynamic}
        for frame_idx in _frame_range(row):
            detailed[str(frame_idx)] = entry
    return dict(sorted(detailed.items(), key=lambda item: int(item[0])))


def convert_prompt_group(
    rows: list[dict[str, str]],
    output_dir: Path,
    id_mode: str = "stem_hash",
    overwrite: bool = False,
) -> PromptConvertResult:
    if not rows:
        raise ValueError("Cannot convert an empty prompt group")

    video_path = rows[0]["path"]
    sample_id = make_sample_id(video_path, id_mode=id_mode)
    json_path = Path(output_dir) / sample_id / "video.json"

    if json_path.exists() and not overwrite:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        detailed = payload.get("detailed", {}) or {}
        return PromptConvertResult(
            sample_id=sample_id,
            path=video_path,
            num_clips=len(rows),
            num_frames=len(detailed),
            json_path=json_path,
        )

    detailed = rows_to_expanded_detailed(rows)
    payload = {"detailed": detailed, "simple": {}}

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return PromptConvertResult(
        sample_id=sample_id,
        path=video_path,
        num_clips=len(rows),
        num_frames=len(detailed),
        json_path=json_path,
    )


def convert_prompt_csv(
    csv_path: Path,
    output_dir: Path,
    id_mode: str = "stem_hash",
    limit_rows: int | None = None,
    limit_videos: int | None = None,
    overwrite: bool = False,
    progress_every: int = 100,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "prompt_manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["sample_id", "path", "num_clips", "num_frames", "json_path"],
        )
        writer.writeheader()

        for index, rows in enumerate(
            iter_prompt_groups(csv_path, limit_rows=limit_rows), start=1
        ):
            if limit_videos is not None and index > limit_videos:
                break

            try:
                result = convert_prompt_group(
                    rows, output_dir, id_mode=id_mode, overwrite=overwrite
                )
            except Exception as exc:
                path = rows[0].get("path", "") if rows else ""
                print(f"[ERROR] video={index} path={path}: {exc}")
                continue

            writer.writerow(
                {
                    "sample_id": result.sample_id,
                    "path": result.path,
                    "num_clips": result.num_clips,
                    "num_frames": result.num_frames,
                    "json_path": str(result.json_path),
                }
            )
            if progress_every > 0 and index % progress_every == 0:
                print(
                    f"[{index}] last={result.sample_id} "
                    f"clips={result.num_clips} frames={result.num_frames}"
                )

    print(f"Done. Manifest saved to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stream Unreal clip caption CSV into per-video video.json prompts."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="runjia.qian/train_data_unrealV4_third_person/Unreal_V4.1_caption_clips.csv",
        help="Input CSV with path,text,start_frame,end_frame columns.",
    )
    parser.add_argument(
        "output_dir",
        help="Output root. Each video writes <output_dir>/<sample_id>/video.json.",
    )
    parser.add_argument(
        "--id-mode",
        default="stem_hash",
        choices=["stem_hash", "stem", "hash", "path"],
        help="How to name each sample directory. Use the same value as csv_camera_to_npz.py.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Only read N CSV rows. Mainly for smoke tests.",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="Only write N videos. Mainly for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing video.json files.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print one progress line every N videos. Use 0 to disable.",
    )
    args = parser.parse_args()

    convert_prompt_csv(
        csv_path=Path(args.csv_path),
        output_dir=Path(args.output_dir),
        id_mode=args.id_mode,
        limit_rows=args.limit_rows,
        limit_videos=args.limit_videos,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
