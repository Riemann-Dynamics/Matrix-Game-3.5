#!/usr/bin/env python3
"""Batch encode videos into Wan VAE latents.

This script is intentionally prompt-free: each output ``.pth`` contains the
encoded latent tensor and lightweight source metadata only.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm


VIDEO_EXTENSIONS = ("mp4", "mov", "avi", "mkv", "webm", "flv", "wmv")


def _split_csv(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    value = value.replace('"', "").replace("“", "").replace("”", "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _round_to_vae_frames(num_frames):
    return ((num_frames - 1) // 4) * 4 + 1


def build_indice_lookup(frame_indices):
    remaining = [int(index) for index in frame_indices]
    indice_lookup = {}
    group_index = 0
    while remaining:
        group_size = 1 if group_index == 0 else 4
        indice_lookup[group_index] = {"indice": remaining[:group_size]}
        remaining = remaining[group_size:]
        group_index += 1
    return indice_lookup


def find_video_files(
    video_path, extensions=VIDEO_EXTENSIONS, recursive=False, max_scan=None
):
    # 用 os.walk / os.scandir 自己遍历，方便在 NFS 上提前停止；
    # 与原先 glob 行为对齐：跳过隐藏文件/目录（以 "." 开头）。
    ext_set = {f".{ext}".lower() for ext in extensions}
    videos = []

    def _is_target(name):
        return not name.startswith(".") and os.path.splitext(name)[1].lower() in ext_set

    def _iter_files():
        if recursive:
            for root, dirs, files in os.walk(video_path):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                for name in sorted(files):
                    if _is_target(name):
                        yield os.path.join(root, name)
        else:
            try:
                with os.scandir(video_path) as it:
                    entries = sorted(it, key=lambda e: e.name)
                    for entry in entries:
                        if entry.is_file() and _is_target(entry.name):
                            yield entry.path
            except OSError:
                return

    for path in _iter_files():
        videos.append(path)
        if max_scan is not None and len(videos) >= max_scan:
            break

    return sorted(dict.fromkeys(videos))


def find_dl3dv_vipe_videos(video_path, max_scan=None):
    # 提前停止版本：在 NFS 上一边走一边收集，达到 max_scan 后立刻退出。
    videos = []
    for root, dirs, files in os.walk(video_path):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if os.path.basename(root) == "rgb" and "video.mp4" in files:
            videos.append(os.path.join(root, "video.mp4"))
            if max_scan is not None and len(videos) >= max_scan:
                break
    return sorted(dict.fromkeys(videos))


def filter_assigned_videos(videos, video_path, assign_videos):
    assigned = _split_csv(assign_videos)
    if assigned is None:
        return videos

    by_name = {os.path.basename(path): path for path in videos}
    by_rel = {os.path.relpath(path, video_path): path for path in videos}
    selected = []
    missing = []
    for item in assigned:
        if os.path.isabs(item) and os.path.exists(item):
            selected.append(item)
        elif item in by_rel:
            selected.append(by_rel[item])
        elif item in by_name:
            selected.append(by_name[item])
        else:
            missing.append(item)
    if missing:
        print(f"Warning: {len(missing)} assigned videos were not found: {missing[:5]}")
    return selected


def split_list_evenly(items, n_parts):
    parts = [[] for _ in range(n_parts)]
    for index, item in enumerate(items):
        parts[index % n_parts].append(item)
    return parts


def detect_gpu_ids():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [item.strip() for item in visible_devices.split(",") if item.strip()]
        if devices:
            return devices
    if torch.cuda.is_available():
        return [str(index) for index in range(torch.cuda.device_count())]
    return []


def output_path_for_video(video_file, video_root, output_path, preserve_structure):
    video_file = os.path.abspath(video_file)
    if preserve_structure:
        relative = os.path.relpath(video_file, video_root)
        relative = os.path.splitext(relative)[0] + ".pth"
        return os.path.join(output_path, relative)
    return os.path.join(output_path, Path(video_file).stem + ".pth")


def dl3dv_scene_hash(video_file):
    video_path = Path(video_file)
    if video_path.name != "video.mp4" or video_path.parent.name != "rgb":
        return None
    return video_path.parent.parent.name


def copied_video_path_for_video(video_file, output_path, dl3dv_vipe_layout=False):
    if not dl3dv_vipe_layout:
        return None
    scene_hash = dl3dv_scene_hash(video_file)
    if scene_hash is None:
        return None
    return os.path.join(output_path, f"{scene_hash}.mp4")


def copy_video_if_needed(video_file, copied_video_path, overwrite=False):
    if copied_video_path is None:
        return
    os.makedirs(os.path.dirname(copied_video_path), exist_ok=True)
    if os.path.exists(copied_video_path) and not overwrite:
        return
    shutil.copy2(video_file, copied_video_path)


from skvideo.io import vwrite


def save_video_frames(video_path, frames, fps=20, crf=18):
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    vwrite(
        video_path,
        list(frames),
        inputdict={"-r": f"{fps}"},
        outputdict={
            "-vcodec": "libx264",
            "-pix_fmt": "yuv420p",
            "-profile:v": "baseline",
            "-level": "3.1",
            "-preset": "medium",
            "-crf": f"{crf}",
            "-movflags": "+faststart",
        },
    )


class VideoLatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        videos,
        video_root,
        output_path,
        height,
        width,
        video_length,
        frame_interval=1,
        preserve_structure=False,
        overwrite=False,
        dl3dv_vipe_layout=False,
        verbose=False,
    ):
        self.videos = list(videos)
        self.video_root = os.path.abspath(video_root)
        self.output_path = os.path.abspath(output_path)
        self.height = int(height)
        self.width = int(width)
        self.video_length = (
            None if video_length is None else _round_to_vae_frames(int(video_length))
        )
        self.frame_interval = int(frame_interval)
        self.preserve_structure = bool(preserve_structure)
        self.overwrite = bool(overwrite)
        self.dl3dv_vipe_layout = bool(dl3dv_vipe_layout)
        self.verbose = bool(verbose)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_file = self.videos[index]
        scene_hash = dl3dv_scene_hash(video_file) if self.dl3dv_vipe_layout else None
        if scene_hash is None:
            save_path = output_path_for_video(
                video_file,
                self.video_root,
                self.output_path,
                preserve_structure=self.preserve_structure,
            )
        else:
            save_path = os.path.join(self.output_path, f"{scene_hash}.pth")
        copied_video_path = copied_video_path_for_video(
            video_file,
            self.output_path,
            dl3dv_vipe_layout=self.dl3dv_vipe_layout,
        )
        if os.path.exists(save_path) and not self.overwrite:
            return {
                "video_path": video_file,
                "save_path": save_path,
                "copied_video_path": copied_video_path or "",
                "verbose_video_path": os.path.splitext(save_path)[0] + ".mp4",
                "skip": True,
            }

        video, frame_indices, verbose_frames = self.load_video(video_file)
        item = {
            "video_path": video_file,
            "save_path": save_path,
            "copied_video_path": copied_video_path or "",
            "verbose_video_path": os.path.splitext(save_path)[0] + ".mp4",
            "video": video,
            "frame_indices": frame_indices,
            "skip": False,
        }
        if verbose_frames is not None:
            item["verbose_frames"] = verbose_frames
        return item

    def load_video(self, video_file):
        from decord import VideoReader, cpu

        reader = VideoReader(str(video_file), ctx=cpu())
        available_indices = list(range(0, len(reader), self.frame_interval))
        if not available_indices:
            raise ValueError(f"No frames found in video: {video_file}")

        if self.video_length is None:
            num_frames = len(available_indices)
        else:
            num_frames = min(self.video_length, len(available_indices))
        num_frames = _round_to_vae_frames(num_frames)
        frame_indices = available_indices[:num_frames]

        frames = reader.get_batch(frame_indices).asnumpy()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        frames = self.center_crop_resize(frames)
        verbose_frames = None
        if self.verbose:
            verbose_frames = (
                (frames * 255.0).clamp(0, 255).permute(0, 2, 3, 1).to(dtype=torch.uint8)
            )
        frames = (frames - 0.5) / 0.5
        frames = rearrange(frames, "T C H W -> C T H W")
        return frames, np.asarray(frame_indices, dtype=np.int64), verbose_frames

    def center_crop_resize(self, frames):
        # 先按 "cover" 方式等比缩放，保证缩放后两个维度都不小于目标尺寸，
        # 再做中心裁剪。例如 16:9 输入在目标 1280x704 时，会先缩放到
        # 1280x720，再上下各裁掉 8 像素，得到 1280x704。
        _, _, height, width = frames.shape
        scale = max(self.height / height, self.width / width)
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))
        if (new_height, new_width) != (height, width):
            frames = F.interpolate(
                frames,
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        top = max((new_height - self.height) // 2, 0)
        left = max((new_width - self.width) // 2, 0)
        frames = frames[:, :, top : top + self.height, left : left + self.width]
        return frames


class WanVideoLatentEncoder:
    def __init__(
        self,
        vae_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        perframe=False,
    ):
        from diffsynth.core import ModelConfig
        from diffsynth.pipelines.wan_video import WanVideoPipeline

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device="cpu",
            model_configs=[ModelConfig(path=vae_path, offload_device="cpu")],
        )
        self.pipe.to(device)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.perframe = perframe
        self.tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

    @torch.no_grad()
    def encode_and_save(self, batch):
        video_path = batch["video_path"][0]
        copied_video_path = batch.get("copied_video_path", [""])[0] or None
        verbose_video_path = batch.get("verbose_video_path", [""])[0] or None
        copy_video_if_needed(video_path, copied_video_path)

        if batch.get("skip", [False])[0]:
            print(f"Skip existing file: {batch['save_path'][0]}")
            return

        save_path = batch["save_path"][0]
        video = batch["video"][0].to(device=self.device, dtype=self.torch_dtype)
        frame_indices = batch["frame_indices"][0].cpu().numpy().astype(np.int64)

        if "verbose_frames" in batch and verbose_video_path:
            verbose_frames = batch["verbose_frames"][0].cpu().numpy()
            save_video_frames(verbose_video_path, verbose_frames, fps=20)
            print(f"Saved verbose video to {verbose_video_path}")

        print(f"Encoding {video_path} -> {save_path}, video shape={tuple(video.shape)}")
        if self.perframe:
            latents = []
            for frame_id in tqdm(range(video.shape[1]), desc="Encoding frames"):
                frame = video[:, frame_id : frame_id + 1, :, :]
                latents.append(
                    self.pipe.vae.encode([frame], device=self.device, tiled=False)[0]
                )
            latents = torch.cat(latents, dim=1)
        else:
            latents = self.pipe.vae.encode(
                [video],
                device=self.device,
                **self.tiler_kwargs,
            )[0]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if self.perframe:
            torch.save(latents.detach().cpu(), save_path)
        else:
            with open(os.path.splitext(save_path)[0] + ".json", "w") as f:
                json.dump(
                    {"indice_lookup": build_indice_lookup(frame_indices)}, f, indent=4
                )
            torch.save(
                {
                    "latents": latents.detach().cpu(),
                    # "frame_indices": frame_indices,
                    # "video_path": video_path,
                    # "copied_video_path": copied_video_path,
                },
                save_path,
            )
        del video, latents


def torch_dtype_from_string(name):
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def launch_workers(args, videos):
    gpu_ids = detect_gpu_ids()
    if args.num_gpus is not None:
        gpu_ids = (
            gpu_ids[: args.num_gpus]
            if gpu_ids
            else [str(i) for i in range(args.num_gpus)]
        )
    if len(gpu_ids) == 0:
        raise RuntimeError("No GPU found. Use --worker --device cpu for CPU debugging.")
    if len(gpu_ids) > len(videos):
        gpu_ids = gpu_ids[: len(videos)]

    splits = split_list_evenly(videos, len(gpu_ids))
    processes = []
    script_path = os.path.abspath(__file__)
    for gpu_id, split in zip(gpu_ids, splits):
        if not split:
            continue
        assigned = ",".join(os.path.relpath(path, args.video_path) for path in split)
        cmd = [
            args.python_exe,
            script_path,
            "--worker",
            "--video_path",
            args.video_path,
            "--output_path",
            args.output_path,
            "--vae_path",
            args.vae_path,
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--frame_interval",
            str(args.frame_interval),
            "--extensions",
            args.extensions,
            "--assign_videos",
            assigned,
            "--device",
            "cuda",
            "--torch_dtype",
            args.torch_dtype,
            "--tile_size_height",
            str(args.tile_size_height),
            "--tile_size_width",
            str(args.tile_size_width),
            "--tile_stride_height",
            str(args.tile_stride_height),
            "--tile_stride_width",
            str(args.tile_stride_width),
            "--dataloader_num_workers",
            str(args.dataloader_num_workers),
        ]
        if args.video_length is not None:
            cmd.extend(["--video_length", str(args.video_length)])
        if args.recursive:
            cmd.append("--recursive")
        if args.preserve_structure:
            cmd.append("--preserve_structure")
        if args.dl3dv_vipe_layout:
            cmd.append("--dl3dv_vipe_layout")
        if args.overwrite:
            cmd.append("--overwrite")
        if args.perframe:
            cmd.append("--perframe")
        if args.verbose:
            cmd.append("--verbose")
        if args.max_scan is not None:
            cmd.extend(["--max_scan", str(args.max_scan)])
        if not args.tiled:
            cmd.append("--no_tiled")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Launch GPU {gpu_id}: {len(split)} videos")
        processes.append(subprocess.Popen(cmd, env=env))

    exit_codes = [process.wait() for process in processes]
    for index, code in enumerate(exit_codes):
        print(f"Worker {index} exit code: {code}")
    if any(code != 0 for code in exit_codes):
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch encode videos to Wan VAE latent .pth files."
    )
    parser.add_argument(
        "--video_path", required=True, help="Directory containing videos."
    )
    parser.add_argument(
        "--output_path", required=True, help="Directory to save .pth files."
    )
    parser.add_argument("--vae_path", required=True, help="Wan VAE checkpoint path.")
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help=(
            "Output frame height. Frames are first resized (cover) so both "
            "dimensions are >= target, then center-cropped. For 16:9 inputs "
            "with default 1280x704, this resizes to 1280x720 and crops 8 px "
            "from top and bottom."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Output frame width. See --height for the resize-then-crop pipeline.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=None,
        help="Maximum frames to read before rounding down to 4N+1. Defaults to full video.",
    )
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--extensions", type=str, default=",".join(VIDEO_EXTENSIONS))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--preserve_structure", action="store_true")
    parser.add_argument(
        "--dl3dv_vipe_layout",
        action="store_true",
        help="Recursively find DL3DV VIPE */rgb/video.mp4 and name outputs by scene hash.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--perframe", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Save processed frames as a same-stem .mp4 next to the latent output.",
    )
    parser.add_argument("--assign_videos", type=str, default=None)
    parser.add_argument(
        "--max_scan",
        type=int,
        default=None,
        help=(
            "Stop scanning the input directory after collecting this many "
            "videos. Useful for debugging when the source lives on a slow "
            "filesystem (e.g. NFS). Default: no limit."
        ),
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--no_tiled", dest="tiled", action="store_false")
    parser.set_defaults(tiled=True)
    parser.add_argument("--tile_size_height", type=int, default=30)
    parser.add_argument("--tile_size_width", type=int, default=52)
    parser.add_argument("--tile_stride_height", type=int, default=15)
    parser.add_argument("--tile_stride_width", type=int, default=26)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.video_path = os.path.abspath(args.video_path)
    args.output_path = os.path.abspath(args.output_path)
    args.dl3dv_vipe_layout = args.dl3dv_vipe_layout or (
        "vipe_outputs" in Path(args.video_path).parts
    )

    extensions = tuple(_split_csv(args.extensions) or VIDEO_EXTENSIONS)
    if args.dl3dv_vipe_layout:
        videos = find_dl3dv_vipe_videos(args.video_path, max_scan=args.max_scan)
    else:
        videos = find_video_files(
            args.video_path,
            extensions,
            recursive=args.recursive,
            max_scan=args.max_scan,
        )
    if args.max_scan is not None:
        print(f"Found {len(videos)} videos (scan capped at --max_scan={args.max_scan})")
    else:
        print(f"Found {len(videos)} videos")
    videos = filter_assigned_videos(videos, args.video_path, args.assign_videos)
    if len(videos) == 0:
        raise RuntimeError(f"No videos found under {args.video_path}.")

    os.makedirs(args.output_path, exist_ok=True)
    if not args.worker and args.num_gpus and args.num_gpus > 1:
        launch_workers(args, videos)
        return

    dataset = VideoLatentDataset(
        videos=videos,
        video_root=args.video_path,
        output_path=args.output_path,
        height=args.height,
        width=args.width,
        video_length=args.video_length,
        frame_interval=args.frame_interval,
        preserve_structure=args.preserve_structure,
        overwrite=args.overwrite,
        dl3dv_vipe_layout=args.dl3dv_vipe_layout,
        verbose=args.verbose,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )
    encoder = WanVideoLatentEncoder(
        vae_path=args.vae_path,
        device=args.device,
        torch_dtype=torch_dtype_from_string(args.torch_dtype),
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        perframe=args.perframe,
    )
    for batch in tqdm(dataloader, desc="Encoding videos"):
        encoder.encode_and_save(batch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
