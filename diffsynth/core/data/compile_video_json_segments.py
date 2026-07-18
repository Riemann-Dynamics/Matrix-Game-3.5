#!/usr/bin/env python3
import argparse
import fnmatch
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass(frozen=True)
class CompileResult:
    input_path: Path
    output_path: Path
    backup_path: Path
    num_frames: int
    num_segments: int
    skipped: bool = False
    error: str = ""


def _frame_items(detailed):
    if not isinstance(detailed, dict):
        raise ValueError(f"Expected detailed to be a dict, got {type(detailed).__name__}")

    items = []
    for key, prompt in detailed.items():
        try:
            frame = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Frame key must be an integer string, got {key!r}") from exc
        items.append((frame, prompt))
    items.sort(key=lambda item: item[0])
    return items


def detailed_to_segments(detailed):
    segments = []
    current_start = None
    current_end = None
    current_prompt = None

    for frame, prompt in _frame_items(detailed):
        starts_new_segment = (
            current_start is None
            or frame != current_end + 1
            or prompt != current_prompt
        )
        if starts_new_segment:
            if current_start is not None:
                segments.append(
                    {
                        "start": current_start,
                        "end": current_end,
                        "prompt": current_prompt,
                    }
                )
            current_start = frame
            current_prompt = prompt
        current_end = frame

    if current_start is not None:
        segments.append(
            {"start": current_start, "end": current_end, "prompt": current_prompt}
        )
    return segments


def backup_path_for(input_path: Path) -> Path:
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.name}.bak")


def compile_video_json(
    input_path: Path,
    *,
    overwrite: bool = False,
) -> CompileResult:
    input_path = Path(input_path)
    output_path = input_path
    backup_path = backup_path_for(input_path)

    if backup_path.exists() and not overwrite:
        return CompileResult(input_path, output_path, backup_path, 0, 0, skipped=True)

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in {input_path}")
    if "detailed" not in payload:
        raise ValueError(f"Missing required key 'detailed' in {input_path}")

    detailed = payload["detailed"]
    segments = detailed_to_segments(detailed)

    output_payload = dict(payload)
    output_payload["detailed"] = segments

    if backup_path.exists() and overwrite:
        backup_path = _next_backup_path(input_path)
    input_path.rename(backup_path)
    with input_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return CompileResult(
        input_path=input_path,
        output_path=output_path,
        backup_path=backup_path,
        num_frames=len(detailed),
        num_segments=len(segments),
    )


def _next_backup_path(input_path: Path) -> Path:
    for index in range(1, 1000000):
        candidate = input_path.with_name(f"{input_path.name}.bak.{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find an available backup path for {input_path}")


def iter_video_json_paths(
    inputs,
    pattern: str = "video.json",
    progress=None,
    recursive: bool = False,
):
    seen = set()
    for raw_path in _expand_input_paths(inputs):
        path = Path(raw_path)
        if path.is_dir():
            if recursive:
                yield from _iter_find_files(path, pattern, seen=seen, progress=progress)
            else:
                yield from _iter_shallow_video_json_candidates(
                    path,
                    pattern,
                    seen=seen,
                    progress=progress,
                )
        else:
            candidate = path
            seen_key = os.path.abspath(os.fspath(candidate))
            if seen_key in seen:
                continue
            seen.add(seen_key)
            yield candidate


def _expand_input_paths(inputs):
    for raw_input in inputs:
        for part in str(raw_input).split(","):
            part = part.strip()
            if part:
                yield part


def _iter_shallow_video_json_candidates(root: Path, pattern: str, *, seen, progress=None):
    if fnmatch.fnmatch("video.json", pattern):
        root_candidate = root / "video.json"
        seen_key = os.path.abspath(os.fspath(root_candidate))
        if root_candidate.exists() and seen_key not in seen:
            seen.add(seen_key)
            if progress is not None:
                progress.update(1)
            yield root_candidate

    try:
        with os.scandir(root) as entries:
            for entry in entries:
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue

                candidate = Path(entry.path) / "video.json"
                if not fnmatch.fnmatch(candidate.name, pattern):
                    continue
                seen_key = os.path.abspath(os.fspath(candidate))
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                if progress is not None:
                    progress.update(1)
                yield candidate
    except OSError:
        return


def _iter_find_files(root: Path, pattern: str, *, seen, progress=None):
    command = ["find", str(root), "-type", "f", "-name", pattern, "-print"]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        yield from _iter_matching_files(
            root, pattern, seen=seen, progress=progress, recursive=True
        )
        return

    assert process.stdout is not None
    for line in process.stdout:
        raw_path = line.rstrip("\n")
        if not raw_path:
            continue
        candidate = Path(raw_path)
        seen_key = os.path.abspath(raw_path)
        if seen_key in seen:
            continue
        seen.add(seen_key)
        if progress is not None:
            progress.update(1)
        yield candidate

    process.wait()
    if process.returncode not in (0, None):
        raise RuntimeError(f"find failed for {root} with exit code {process.returncode}")


def _iter_matching_files(root: Path, pattern: str, *, seen, progress=None, recursive=True):
    stack = [Path(root)]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if recursive and entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(
                            follow_symlinks=False
                        ) and fnmatch.fnmatch(entry.name, pattern):
                            candidate = Path(entry.path)
                            seen_key = os.path.abspath(entry.path)
                            if seen_key in seen:
                                continue
                            seen.add(seen_key)
                            if progress is not None:
                                progress.update(1)
                            yield candidate
                    except OSError:
                        continue
        except OSError:
            continue


def _compile_worker(args):
    input_path, overwrite = args
    try:
        return compile_video_json(
            input_path,
            overwrite=overwrite,
        )
    except Exception as exc:
        input_path = Path(input_path)
        return CompileResult(
            input_path,
            input_path,
            backup_path_for(input_path),
            0,
            0,
            error=str(exc),
        )


def compile_many_video_jsons(
    input_paths,
    *,
    overwrite: bool = False,
    workers: int | None = None,
):
    paths = [Path(path) for path in input_paths]
    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, int(workers))

    if workers == 1:
        for path in paths:
            yield _compile_worker((path, overwrite))
        return

    tasks = [(path, overwrite) for path in paths]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_compile_worker, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


def _write_manifest(manifest_path: Path, results):
    import csv

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_path",
                "output_path",
                "backup_path",
                "num_frames",
                "num_segments",
                "skipped",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "input_path": str(result.input_path),
                    "output_path": str(result.output_path),
                    "backup_path": str(result.backup_path),
                    "num_frames": result.num_frames,
                    "num_segments": result.num_segments,
                    "skipped": int(result.skipped),
                    "error": result.error,
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compile expanded per-frame video.json prompts into contiguous "
            "prompt segments in place. Each source video.json is renamed to "
            "video.json.bak before the compact video.json is written."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input video.json files or directories to scan recursively.",
    )
    parser.add_argument(
        "--pattern",
        default="video.json",
        help="Filename pattern used when an input is a directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Recursively scan all descendant directories. By default directory "
            "inputs only scan root/video.json and root/*/video.json for speed."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Process files even if video.json.bak exists. Existing backups are "
            "preserved by writing video.json.bak.N."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional CSV manifest path summarizing conversion results.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N finished files. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    args = parser.parse_args()

    scan_progress = None
    if not args.no_tqdm:
        scan_progress = tqdm(
            total=None,
            desc="Finding video.json",
            unit="file",
        )
    try:
        paths = list(
            iter_video_json_paths(
                args.inputs,
                pattern=args.pattern,
                progress=scan_progress,
                recursive=args.recursive,
            )
        )
    finally:
        if scan_progress is not None:
            scan_progress.close()
    if not paths:
        raise SystemExit("No input JSON files found.")

    result_iter = compile_many_video_jsons(
        paths,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    if not args.no_tqdm:
        result_iter = tqdm(
            result_iter,
            total=len(paths),
            desc="Compiling video.json",
            unit="file",
        )

    results = []
    for index, result in enumerate(result_iter, start=1):
        results.append(result)
        if result.error:
            print(f"[ERROR] {result.input_path}: {result.error}")
        elif args.progress_every > 0 and index % args.progress_every == 0:
            print(f"[{index}/{len(paths)}] last={result.input_path}")

    if args.manifest:
        _write_manifest(Path(args.manifest), results)

    failed = sum(1 for result in results if result.error)
    skipped = sum(1 for result in results if result.skipped)
    written = len(results) - failed - skipped
    print(
        f"Done. total={len(results)} written={written} "
        f"skipped={skipped} failed={failed}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
