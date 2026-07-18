from .common import *
from .cleanup import _release_cached_memory
from .history import _select_validation_clean_latents
from .timing import NOOP_TIMING_REPORTER
from .video_io import _read_video_gt_frames

VALIDATION_QWEN_MODES = (
    "oracle_section_gt",
    "oracle_full_gt_static",
    "self_rollout", 
    "self_rollout_event",
)


DEFAULT_VALIDATION_QWEN_EVENTS = (
    {
        "id": "red_sports_car",
        "text": "a red sports car enters from the right and becomes part of the scene",
        "tags": ["vehicle"],
        "phase_hint": "enter_then_continue",
    },
    {
        "id": "astronaut_right",
        "text": "an astronaut walks into the frame",
        "tags": ["person"],
        "phase_hint": "enter_then_continue",
    },
    {
        "id": "knight_walks",
        "text": "a medieval knight was seen entering the frame",
        "tags": ["person"],
        "phase_hint": "enter_then_continue",
    },
    {
        "id": "armored_vehicle",
        "text": "a armored vehicle runs into the frame",
        "tags": ["vehicle"],
        "phase_hint": "enter_then_continue",
    },
    {
        "id": "hot_lava",
        "text": "hot lava flows into the scene from the right",
        "tags": ["weather"],
        "phase_hint": "enter_then_continue",
    },
    {
        "id": "a_black_horse",
        "text": "a black horse walks into the scene from the left",
        "tags": ["animal"],
        "phase_hint": "enter_then_continue",
    },
    {
        "id": "a_white_bear",
        "text": "a white bear walks into the scene from the left",
        "tags": ["animal"],
        "phase_hint": "enter_then_continue",
    },
    {
        "id": "a_portal_opens",
        "text": "a portal opens in the scene and an alien environment like mars surface can be seen through the portal",
        "tags": ["portal"],
        "phase_hint": "enter_then_continue",
    },
)


def _normalize_validation_qwen_mode(mode):
    mode = str(mode or "oracle_section_gt").strip().lower()
    aliases = {
        "section_gt": "oracle_section_gt",
        "gt_section": "oracle_section_gt",
        "full_gt": "oracle_full_gt_static",
        "full_gt_static": "oracle_full_gt_static",
        "static_gt": "oracle_full_gt_static",
        "rollout": "self_rollout",
        "self": "self_rollout",
        "interactive": "self_rollout_event",
        "interactive_event": "self_rollout_event",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALIDATION_QWEN_MODES:
        raise ValueError(
            "validation_qwen_mode must be one of "
            f"{VALIDATION_QWEN_MODES}, got {mode!r}."
        )
    return mode


def _normalize_validation_qwen_event_entry(entry, index):
    if isinstance(entry, str):
        text = entry.strip()
        if not text:
            return None
        return {
            "id": f"event_{int(index):03d}",
            "text": text,
            "tags": [],
            "phase_hint": "",
        }
    if not isinstance(entry, dict):
        raise ValueError(
            "validation_qwen_event_yaml entries must be strings or mappings; "
            f"got {type(entry).__name__}."
        )
    text = str(
        entry.get("text")
        or entry.get("prompt")
        or entry.get("description")
        or entry.get("event")
        or ""
    ).strip()
    if not text:
        raise ValueError("validation_qwen_event_yaml entry is missing text.")
    tags = entry.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return {
        "id": str(entry.get("id") or f"event_{int(index):03d}").strip(),
        "text": text,
        "tags": [str(tag) for tag in tags],
        "phase_hint": str(entry.get("phase_hint") or "").strip(),
    }


def _load_validation_qwen_events(path):
    path = str(path or "").strip()
    if not path:
        return [
            _normalize_validation_qwen_event_entry(entry, idx)
            for idx, entry in enumerate(DEFAULT_VALIDATION_QWEN_EVENTS)
        ]
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or []
    if isinstance(payload, dict):
        payload = payload.get("events") or payload.get("validation_events") or []
    if not isinstance(payload, list):
        raise ValueError(
            "validation_qwen_event_yaml must be a list or a mapping with an events list."
        )
    events = []
    for idx, entry in enumerate(payload):
        event = _normalize_validation_qwen_event_entry(entry, idx)
        if event is not None:
            events.append(event)
    if not events:
        raise ValueError("validation_qwen_event_yaml did not contain any events.")
    return events


def _select_validation_qwen_event(events, *, sample_index, seed):
    events = list(events or [])
    if not events:
        raise ValueError("self_rollout_event requires at least one validation event.")
    order = list(events)
    rng = random.Random((int(seed or 0) << 17) ^ 0x5EED17)
    rng.shuffle(order)
    return dict(order[int(sample_index) % len(order)])


def _collect_validation_model_config_text(model_path):
    model_path = str(model_path or "").strip()
    texts = [model_path, os.path.basename(model_path)]
    if model_path and os.path.isdir(model_path):
        for filename in (
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
        ):
            path = os.path.join(model_path, filename)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                for key in (
                    "model_type",
                    "architectures",
                    "_name_or_path",
                    "name_or_path",
                    "processor_class",
                ):
                    value = payload.get(key)
                    if value is not None:
                        texts.append(json.dumps(value))
            except (OSError, ValueError):
                continue
    return " ".join(str(text) for text in texts).lower()


def _is_supported_validation_gemma4_model_text(text):
    normalized = str(text or "").lower().replace("_", "-")
    return "gemma-4-e2b-it" in normalized or "gemma-4-e4b-it" in normalized


def _validation_gemma4_unsupported_error(mode="validation"):
    option_prefix = "train" if str(mode) == "train" else "validation"
    return ValueError(
        f"{option_prefix}_qwen_provider=gemma4 currently supports only "
        "gemma-4-E2B-it or gemma-4-E4B-it local models. "
        f"Use a supported local model path or pass --{option_prefix}_qwen_provider "
        "qwen for Qwen models."
    )


def _resolve_validation_qwen_provider(provider, model_path, *, mode="validation"):
    provider = str(provider or "auto").strip().lower()
    if provider not in {"auto", "qwen", "gemma4"}:
        option_prefix = "train" if str(mode) == "train" else "validation"
        raise ValueError(
            f"{option_prefix}_qwen_provider must be one of "
            f"{{'auto', 'qwen', 'gemma4'}}, got {provider!r}."
        )
    text = _collect_validation_model_config_text(model_path)
    if provider == "gemma4":
        if not _is_supported_validation_gemma4_model_text(text):
            raise _validation_gemma4_unsupported_error(mode)
        return provider
    if provider != "auto":
        return provider
    if _is_supported_validation_gemma4_model_text(text):
        return "gemma4"
    if "gemma" in text:
        raise _validation_gemma4_unsupported_error(mode)
    if "qwen" in text:
        return "qwen"
    option_prefix = "train" if str(mode) == "train" else "validation"
    raise ValueError(
        f"Unable to infer {option_prefix} Qwen prompt provider from model "
        f"path/config. Pass --{option_prefix}_qwen_provider qwen or "
        f"--{option_prefix}_qwen_provider gemma4."
    )


def _validation_qwen_extrinsics_to_numpy(extrinsics):
    if torch.is_tensor(extrinsics):
        return extrinsics.detach().cpu().numpy()
    return np.asarray(extrinsics)


def _summarize_validation_camera_motion(extrinsics):
    arr = np.asarray(extrinsics, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] <= 1 or arr.shape[1] < 3 or arr.shape[2] < 4:
        return "camera motion is unavailable"
    start = arr[0, :3, 3]
    end = arr[-1, :3, 3]
    delta = end - start
    distance = float(np.linalg.norm(delta))
    parts = []
    eps = 1e-3
    if abs(float(delta[2])) > eps:
        parts.append("forward" if float(delta[2]) < 0 else "backward")
    if abs(float(delta[0])) > eps:
        parts.append("right" if float(delta[0]) > 0 else "left")
    if abs(float(delta[1])) > eps:
        parts.append("up" if float(delta[1]) > 0 else "down")
    rotation_note = ""
    try:
        rel_rot = arr[-1, :3, :3] @ arr[0, :3, :3].T
        trace = float(np.trace(rel_rot))
        angle = float(np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))))
        if angle > 3.0:
            rotation_note = f" while turning about {angle:.0f} degrees"
    except Exception:
        rotation_note = ""
    if not parts and not rotation_note:
        return "camera remains mostly static"
    if not parts:
        movement = "camera mostly rotates"
    else:
        movement = "camera moves " + " and ".join(parts)
        if distance > 0:
            movement += f" over a {distance:.2f} pose-unit path"
    return movement + rotation_note


def _qwen_video_prompt_instruction(
    image_count,
    word_limit=200,
    *,
    prompt_mode="oracle_section_gt",
    event=None,
    camera_summary="",
    dataset_prompt="",
    section_idx=0,
    total_sections=None,
    image_source="",
):
    word_limit = max(1, int(word_limit or 200))
    image_count = max(1, int(image_count or 1))
    prompt_mode = _normalize_validation_qwen_mode(prompt_mode)
    section_text = (
        f" This is section {int(section_idx) + 1} of {int(total_sections)}."
        if total_sections is not None
        else ""
    )
    context_bits = []
    dataset_prompt = str(dataset_prompt or "").strip()
    if dataset_prompt:
        context_bits.append(f"Scene anchor from dataset caption: {dataset_prompt}")
    camera_summary = str(camera_summary or "").strip()
    if camera_summary:
        context_bits.append(f"Actual validation camera context: {camera_summary}.")
    if event is not None:
        event_text = event.get("text") if isinstance(event, dict) else str(event)
        phase_hint = event.get("phase_hint", "") if isinstance(event, dict) else ""
        event_clause = f"User event for this rollout: {str(event_text).strip()}."
        if phase_hint:
            event_clause += f" Event phase hint: {phase_hint}."
        context_bits.append(event_clause)
    context = " ".join(context_bits)
    if context:
        context = " " + context
    source_notes = {
        "oracle_section_gt": (
            "The images are sampled from the future ground-truth segment that "
            "will be generated."
        ),
        "oracle_full_gt_static": (
            "The images are sampled across the full ground-truth rollout. "
            "Write one global prompt that can be reused for every segment."
        ),
        "self_rollout": (
            "The images show only the current or past autoregressive rollout "
            "context. Write the next segment prompt without relying on future frames."
        ),
        "self_rollout_event": (
            "The images show only the current or past autoregressive rollout "
            "context. Write the next segment prompt while carrying the user event "
            "forward naturally; do not reintroduce the same event as a brand-new "
            "arrival in every section."
        ),
    }
    return (
        f"{source_notes[prompt_mode]}{section_text}{context} "
        f"Describe the video generation prompt in coherent English around "
        f"{word_limit} words. Focus on scene, objects, object appearance, "
        "materials, readable text if visible, spatial relationships, actions, "
        "atmosphere, lighting, and the event state when provided. Brief camera "
        "motion wording is allowed when it helps match the actual camera context. "
        "Avoid starting with phrases such as 'in the picture' or 'this is a picture'. "
        "Do not include reasoning, bullet points, markdown, labels, or instructions. "
        "Return only the final prompt in one paragraph without line breaks."
    )


def _validation_qwen_prompt_instruction(word_limit=200, image_count=1):
    return _qwen_video_prompt_instruction(image_count, word_limit=word_limit)


def _sanitize_validation_qwen_prompt(text, word_limit=200):
    text = "" if text is None else str(text)
    text = html.unescape(text)
    text = re.sub(r"(?is)<think>.*?</think>", " ", text)

    lines = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        lines.append(line)

    text = " ".join(lines)
    text = re.sub(r"[*_`#>\[\]]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" \t\r\n\"'`")

    prefix_pattern = re.compile(
        r"^(?:final\s+)?(?:english\s+)?(?:video[-\s]generation\s+)?"
        r"(?:prompt|caption|description)\s*[:：\-]\s*",
        flags=re.IGNORECASE,
    )
    while True:
        stripped = prefix_pattern.sub("", text).strip()
        if stripped == text:
            break
        text = stripped

    word_limit = max(1, int(word_limit or 200))
    words = text.split()
    if len(words) > word_limit:
        text = " ".join(words[:word_limit]).rstrip(" ,;:")
    return text


def _validation_qwen_torch_dtype(dtype_name):
    name = str(dtype_name or "auto").strip().lower()
    if name in ("auto", "none"):
        return "auto"
    attr_by_name = {
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float16": "float16",
        "fp16": "float16",
        "float32": "float32",
        "fp32": "float32",
    }
    attr = attr_by_name.get(name)
    if attr is None:
        raise ValueError(f"Unsupported --validation_qwen_torch_dtype: {dtype_name!r}")
    if not hasattr(torch, attr):
        raise ValueError(f"Current torch build has no dtype torch.{attr}")
    return getattr(torch, attr)


def _qwen_model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _qwen_model_to_device(model, device):
    if model is None or device is None:
        return
    if hasattr(model, "to"):
        model.to(device)


def _move_qwen_inputs_to_device(inputs, device):
    if device is None:
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    return inputs


def _qwen_inputs_input_ids(inputs):
    input_ids = getattr(inputs, "input_ids", None)
    if input_ids is not None:
        return input_ids
    if isinstance(inputs, dict):
        return inputs.get("input_ids")
    return None


def _sample_train_qwen_frame_indices(total_frames, sample_frames):
    total_frames = int(total_frames)
    sample_frames = int(sample_frames)
    if total_frames <= 0 or sample_frames <= 0:
        return []
    count = min(total_frames, sample_frames)
    if count == 1:
        return [0]
    raw = np.linspace(0, total_frames - 1, count)
    indices = []
    seen = set()
    for value in raw:
        idx = int(round(float(value)))
        idx = max(0, min(total_frames - 1, idx))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    return indices


def _validation_qwen_context_frame_indices(total_frames, frame_count):
    total_frames = int(total_frames)
    frame_count = max(1, int(frame_count or 1))
    if total_frames <= 0:
        return []
    if frame_count == 1:
        return [total_frames - 1]
    raw = np.linspace(0, total_frames - 1, frame_count)
    return [max(0, min(total_frames - 1, int(round(float(value))))) for value in raw]


def _validation_qwen_section_frame_indices(
    info,
    *,
    section_idx,
    latent_window_size,
    sample_frames,
):
    info = info or {}
    section_idx = max(0, int(section_idx))
    latent_window_size = max(1, int(latent_window_size))
    section_frame_count = 4 * latent_window_size
    start = int(info.get("generating_first_frame_idx", 0) or 0)
    start += section_idx * section_frame_count
    local_indices = _sample_train_qwen_frame_indices(
        section_frame_count,
        sample_frames,
    )
    return [int(start + idx) for idx in local_indices]


def _validation_qwen_full_rollout_frame_indices(
    info,
    *,
    num_sections,
    latent_window_size,
    sample_frames,
):
    info = info or {}
    num_sections = max(1, int(num_sections or 1))
    latent_window_size = max(1, int(latent_window_size))
    total_frame_count = 4 * latent_window_size * num_sections
    start = int(info.get("generating_first_frame_idx", 0) or 0)
    local_indices = _sample_train_qwen_frame_indices(total_frame_count, sample_frames)
    return [int(start + idx) for idx in local_indices]


def _validation_qwen_full_rollout_camera_summary(
    val_dataset,
    info,
    *,
    num_sections,
    latent_window_size,
):
    info = info or {}
    try:
        extrinsics = val_dataset._load_extrinsics(val_dataset._get_extrinsic_path(info))
    except Exception:
        return ""
    frame_indices = _validation_qwen_full_rollout_frame_indices(
        info,
        num_sections=num_sections,
        latent_window_size=latent_window_size,
        sample_frames=4 * int(latent_window_size) * max(1, int(num_sections or 1)),
    )
    if not frame_indices:
        return ""
    max_idx = int(extrinsics.shape[0]) - 1
    frame_indices = [min(max(0, int(idx)), max_idx) for idx in frame_indices]
    return _summarize_validation_camera_motion(extrinsics[frame_indices])


def _validation_qwen_actual_camera_summaries(
    val_dataset,
    data,
    *,
    num_sections,
    latent_window_size,
    history_latents,
    validation_clean_latent_history_count,
    camera_pingpong=False,
):
    num_sections = max(1, int(num_sections or 1))
    latent_window_size = max(1, int(latent_window_size or 1))
    running_clean_indices = data["clean_latent_indices"].clone()
    running_noisy_indices = data["noisy_latent_indices"].clone()
    summaries = []
    for section_idx in range(num_sections):
        indice_drift = latent_window_size if section_idx > 0 else 0
        running_clean_indices = running_clean_indices + indice_drift
        running_noisy_indices = running_noisy_indices + indice_drift
        current_clean_latents = _select_validation_clean_latents(
            history_latents,
            section_idx=section_idx,
            requested_history_count=validation_clean_latent_history_count,
        )
        camera_data = val_dataset.read_camera_params(
            {
                "clean_latent_indices_start": data.get("clean_latent_indices_start"),
                "clean_latent_indices": running_clean_indices,
                "noisy_latent_indices": running_noisy_indices,
                "clean_latents": current_clean_latents,
                "info": data.get("info", {}),
                "lookup": data.get("lookup", {}),
                "is_starting": (section_idx == 0),
                # Mirror the rollout's camera index mapping so the Qwen prompt
                # describes the SAME trajectory PRoPE actually conditions on
                # (validation.py threads camera_pingpong into its own
                # read_camera_params call too); otherwise --validation_camera_
                # pingpong would freeze the prompt on the last pose while the
                # generation keeps ping-ponging.
                "camera_pingpong": bool(camera_pingpong),
            },
            info=data.get("info", {}),
            lookup=data.get("lookup", {}),
        )
        summaries.append(
            _summarize_validation_camera_motion(
                _validation_qwen_extrinsics_to_numpy(
                    camera_data["noisy_latent_indices_prope_extrinsic"]
                )
            )
        )
    return summaries


def _pil_downsample_max_side(image, max_side):
    max_side = int(max_side or 0)
    if max_side <= 0:
        return image
    width, height = image.size
    longer = max(int(width), int(height))
    if longer <= max_side:
        return image
    scale = float(max_side) / float(longer)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _train_qwen_frames_from_cn_frames(cn_frames, sample_frames, max_side):
    if not torch.is_tensor(cn_frames):
        raise TypeError("cn_frames must be a torch.Tensor")
    if cn_frames.ndim != 4:
        raise ValueError(
            "cn_frames must have shape (C,T,H,W), " f"got {tuple(cn_frames.shape)}."
        )
    if int(cn_frames.shape[0]) != 3:
        raise ValueError(
            "cn_frames must have three RGB channels, "
            f"got C={int(cn_frames.shape[0])}."
        )
    indices = _sample_train_qwen_frame_indices(cn_frames.shape[1], sample_frames)
    images = []
    for idx in indices:
        frame = cn_frames[:, int(idx)].detach().cpu()
        if frame.dtype == torch.uint8:
            arr = frame[:3].permute(1, 2, 0).numpy().astype(np.uint8, copy=False)
        else:
            frame = frame.float()
            arr_f = frame.permute(1, 2, 0)
            if bool(arr_f.numel()) and float(arr_f.max().item()) > 2.0:
                arr_f = arr_f
            elif bool(arr_f.numel()) and float(arr_f.min().item()) >= 0.0:
                arr_f = arr_f * 255.0
            else:
                arr_f = (arr_f + 1.0) * 127.5
            arr = arr_f.clamp(0, 255).numpy().astype(np.uint8)
        image = Image.fromarray(arr).convert("RGB")
        images.append(_pil_downsample_max_side(image, max_side))
    return images, indices


def _numeric_json_key_sort_key(key):
    try:
        return (0, int(key))
    except (TypeError, ValueError):
        return (1, str(key))


def _safe_train_qwen_prompt_cache_scene_id(scene_id):
    text = str(scene_id or "").strip()
    if not text:
        text = "unknown"
    text = text.replace(os.sep, "_")
    if os.altsep:
        text = text.replace(os.altsep, "_")
    return text


def _normalize_train_qwen_prompt_cache_frame_indices(frame_indices):
    values = []
    for value in frame_indices or []:
        try:
            frame_id = int(value)
        except (TypeError, ValueError):
            continue
        if frame_id < 0:
            continue
        values.append(frame_id)
    return sorted(dict.fromkeys(values))


def _write_train_qwen_prompt_cache(cache_dir, scene_id, frame_indices, prompt):
    cache_dir = str(cache_dir or "").strip()
    if not cache_dir:
        raise ValueError("train Qwen prompt cache_dir must be non-empty.")
    scene_id = _safe_train_qwen_prompt_cache_scene_id(scene_id)
    frames = _normalize_train_qwen_prompt_cache_frame_indices(frame_indices)
    if not frames:
        raise ValueError("train Qwen prompt cache write requires frame indices.")

    os.makedirs(cache_dir, exist_ok=True)
    target_path = os.path.join(cache_dir, f"{scene_id}.json")
    lock_path = os.path.join(cache_dir, f".{scene_id}.json.lock")
    prompt = str(prompt or "")

    with open(lock_path, "a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = {}
            if os.path.exists(target_path):
                try:
                    with open(target_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        payload = loaded
                except (OSError, ValueError):
                    payload = {}
            detailed = payload.get("detailed", {})
            if not isinstance(detailed, dict):
                detailed = {}
            entry = {"start": "", "dynamic": prompt}
            for frame_id in frames:
                detailed[str(int(frame_id))] = dict(entry)
            payload["prompt_cache_format"] = "frame_entries_v1"
            payload["detailed"] = {
                key: detailed[key]
                for key in sorted(detailed.keys(), key=_numeric_json_key_sort_key)
            }

            encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8"
            )
            if os.path.exists(target_path):
                old_size = os.path.getsize(target_path)
                with open(target_path, "r+b") as f:
                    f.write(encoded)
                    if len(encoded) < old_size:
                        f.write(b" " * (old_size - len(encoded)))
            else:
                with open(target_path, "xb") as f:
                    f.write(encoded)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return {
        "path": target_path,
        "scene_id": scene_id,
        "frame_count": int(len(frames)),
        "frame_min": int(frames[0]),
        "frame_max": int(frames[-1]),
    }


def _pil_images_from_uint8_frames(frames_np, max_side):
    frames_arr = np.asarray(frames_np)
    if frames_arr.ndim != 4 or frames_arr.shape[-1] != 3:
        raise ValueError(
            "_pil_images_from_uint8_frames expects (T,H,W,3) uint8 frames, "
            f"got shape {tuple(frames_arr.shape)}."
        )
    images = []
    for frame in frames_arr:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        images.append(_pil_downsample_max_side(image, max_side))
    return images


def _validation_qwen_history_prompt_frames(
    frames_np,
    *,
    frame_count,
    frame_max_side,
):
    frames_arr = np.asarray(frames_np)
    if frames_arr.ndim != 4 or frames_arr.shape[-1] != 3:
        raise ValueError(
            "_validation_qwen_history_prompt_frames expects "
            f"(T,H,W,3) frames, got shape {tuple(frames_arr.shape)}."
        )
    indices = _validation_qwen_context_frame_indices(
        frames_arr.shape[0],
        frame_count,
    )
    if not indices:
        return {"frame_indices": [], "images": []}
    selected = frames_arr[indices]
    return {
        "frame_indices": [int(idx) for idx in indices],
        "images": _pil_images_from_uint8_frames(selected, frame_max_side),
    }


def _prepare_validation_qwen_video_prompt_frames(
    info,
    *,
    num_sections,
    latent_window_size,
    sample_frames,
    frame_max_side,
    target_h,
    target_w,
):
    info = info or {}
    video_path = info.get("video_path")
    if not video_path:
        raise ValueError(
            "--validation_qwen_prompt requires data['info']['video_path'] so "
            "validation can sample section-local GT video frames for Qwen."
        )
    by_section = {}
    all_indices = []
    seen = set()
    for section_idx in range(max(0, int(num_sections))):
        frame_indices = _validation_qwen_section_frame_indices(
            info,
            section_idx=section_idx,
            latent_window_size=latent_window_size,
            sample_frames=sample_frames,
        )
        by_section[int(section_idx)] = {
            "frame_indices": frame_indices,
            "images": [],
        }
        for idx in frame_indices:
            idx = int(idx)
            if idx not in seen:
                all_indices.append(idx)
                seen.add(idx)
    if not all_indices:
        return by_section

    frames = _read_video_gt_frames(video_path, all_indices, target_h, target_w)
    if int(frames.shape[0]) != len(all_indices):
        raise RuntimeError(
            "validation Qwen GT frame read returned an unexpected frame count: "
            f"requested={len(all_indices)} got={int(frames.shape[0])}."
        )
    images_by_index = {
        int(idx): image
        for idx, image in zip(
            all_indices,
            _pil_images_from_uint8_frames(frames, frame_max_side),
        )
    }
    for section_payload in by_section.values():
        section_payload["images"] = [
            images_by_index[int(idx)] for idx in section_payload["frame_indices"]
        ]
    return by_section


def _prepare_validation_qwen_full_rollout_prompt_frames(
    info,
    *,
    num_sections,
    latent_window_size,
    sample_frames,
    frame_max_side,
    target_h,
    target_w,
):
    info = info or {}
    video_path = info.get("video_path")
    if not video_path:
        raise ValueError(
            "--validation_qwen_prompt requires data['info']['video_path'] so "
            "validation can sample full-rollout GT video frames for Qwen."
        )
    frame_indices = _validation_qwen_full_rollout_frame_indices(
        info,
        num_sections=num_sections,
        latent_window_size=latent_window_size,
        sample_frames=sample_frames,
    )
    if not frame_indices:
        return {"frame_indices": [], "images": []}
    frames = _read_video_gt_frames(video_path, frame_indices, target_h, target_w)
    if int(frames.shape[0]) != len(frame_indices):
        raise RuntimeError(
            "validation Qwen full-rollout GT frame read returned an unexpected "
            f"frame count: requested={len(frame_indices)} got={int(frames.shape[0])}."
        )
    return {
        "frame_indices": [int(idx) for idx in frame_indices],
        "images": _pil_images_from_uint8_frames(frames, frame_max_side),
    }


def _validation_qwen_context_prompt_frames(frames_np, *, frame_count, frame_max_side):
    frames_arr = np.asarray(frames_np)
    if frames_arr.ndim != 4 or frames_arr.shape[-1] != 3:
        raise ValueError(
            "_validation_qwen_context_prompt_frames expects (T,H,W,3) frames, "
            f"got shape {tuple(frames_arr.shape)}."
        )
    indices = _validation_qwen_context_frame_indices(
        frames_arr.shape[0],
        frame_count,
    )
    if not indices:
        return {"frame_indices": [], "images": []}
    return {
        "frame_indices": [int(idx) for idx in indices],
        "images": _pil_images_from_uint8_frames(frames_arr[indices], frame_max_side),
    }


def _validation_qwen_initial_context_latents(
    history_latents, *, requested_history_count
):
    requested_history_count = max(1, int(requested_history_count or 1))
    count = min(requested_history_count, int(history_latents.shape[2]))
    return history_latents[:, :, -count:].clone()


def _validation_qwen_generate_from_payload(
    qwen_prompt_generator,
    payload,
    *,
    batch_idx,
    section_idx,
    clean_name,
    dataset_prompt,
    timing_reporter,
    transient_cuda_device,
):
    images = list((payload or {}).get("images") or [])
    if not images:
        raise RuntimeError(
            f"[validation][qwen] no prompt frames sampled for section {section_idx}."
        )
    return qwen_prompt_generator.generate_prompt(
        images,
        batch_idx=batch_idx,
        section_idx=section_idx,
        clean_name=clean_name,
        dataset_prompt=dataset_prompt,
        timing_reporter=timing_reporter,
        transient_cuda_device=transient_cuda_device,
        prompt_mode=payload.get("mode", "oracle_section_gt"),
        event=payload.get("event"),
        camera_summary=payload.get("camera_summary", ""),
        total_sections=payload.get("total_sections"),
        image_source=payload.get("image_source", ""),
    )


def _simulate_validation_qwen_prompt_mode(
    *,
    mode,
    num_sections,
    provider_fn,
    gt_payloads,
    recent_payloads,
    camera_summaries,
    event,
    dataset_prompt,
    clean_name,
):
    mode = _normalize_validation_qwen_mode(mode)
    prompts = []
    if mode == "oracle_full_gt_static":
        payload = dict((gt_payloads or {}).get("full") or {})
        payload.update(
            {
                "mode": mode,
                "section_idx": 0,
                "total_sections": int(num_sections),
                "image_source": "gt_full_rollout",
                "camera_summary": " ".join(str(v) for v in camera_summaries or []),
                "event": event,
                "dataset_prompt": dataset_prompt,
                "clean_name": clean_name,
            }
        )
        prompt = provider_fn(payload)
        return [prompt for _ in range(int(num_sections))]
    for section_idx in range(int(num_sections)):
        if mode == "oracle_section_gt":
            payload = dict((gt_payloads or {}).get(section_idx) or {})
            image_source = "gt_section"
        else:
            payload = dict((recent_payloads or {}).get(section_idx) or {})
            image_source = "rollout_context"
        payload.update(
            {
                "mode": mode,
                "section_idx": int(section_idx),
                "total_sections": int(num_sections),
                "image_source": image_source,
                "camera_summary": (
                    list(camera_summaries or [""])[section_idx]
                    if section_idx < len(camera_summaries or [])
                    else ""
                ),
                "event": event if mode == "self_rollout_event" else None,
                "dataset_prompt": dataset_prompt,
                "clean_name": clean_name,
            }
        )
        prompts.append(provider_fn(payload))
    return prompts


class _QwenPromptGenerator:
    def __init__(self, args, *, mode, rank=0):
        self.mode = str(mode)
        if self.mode not in {"validation", "train"}:
            raise ValueError(f"Unsupported Qwen prompt mode: {mode!r}")
        fallback_model_path = getattr(args, "validation_qwen_model_path", "")
        self.model_path = str(
            getattr(args, f"{self.mode}_qwen_model_path", "")
            or fallback_model_path
            or getattr(args, "validation_qwen_model_id", "")
        ).strip()
        self.device_map = str(
            getattr(args, f"{self.mode}_qwen_device_map", "auto") or ""
        ).strip()
        self.torch_dtype_name = str(
            getattr(args, f"{self.mode}_qwen_torch_dtype", "bfloat16") or "auto"
        )
        self.max_new_tokens = max(
            1,
            int(getattr(args, f"{self.mode}_qwen_max_new_tokens", 256) or 256),
        )
        self.word_limit = max(
            1,
            int(getattr(args, f"{self.mode}_qwen_word_limit", 200) or 200),
        )
        self.provider = _resolve_validation_qwen_provider(
            getattr(args, f"{self.mode}_qwen_provider", None) or "auto",
            self.model_path,
            mode=self.mode,
        )
        self.trust_remote_code = bool(
            getattr(args, f"{self.mode}_qwen_trust_remote_code", False)
        )
        self.rank = int(rank)
        self.processor = None
        self.model = None

    def _log(self, message):
        print(
            f"[{self.mode}][qwen:{self.provider}][rank {self.rank}] {message}",
            flush=True,
        )

    def _device_map_name(self):
        return self.device_map.strip().lower()

    def _uses_transient_cuda(self):
        return self._device_map_name() in {
            "transient_cuda",
            "transient-cuda",
            "cuda_transient",
            "gpu_transient",
        }

    def _uses_explicit_cpu(self):
        return self._device_map_name() == "cpu"

    def _model_load_kwargs(self):
        dtype = _validation_qwen_torch_dtype(self.torch_dtype_name)
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        device_map_name = self._device_map_name()
        if self.device_map and device_map_name not in {
            "none",
            "off",
            "false",
            "cpu",
            "transient_cuda",
            "transient-cuda",
            "cuda_transient",
            "gpu_transient",
        }:
            load_kwargs["device_map"] = self.device_map
        return load_kwargs

    def _transient_cuda_target_device(self, fallback_device=None):
        if fallback_device is not None:
            return fallback_device
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    def _load_once(self):
        if self.model is not None and self.processor is not None:
            return
        if not self.model_path:
            raise ValueError(f"--{self.mode}_qwen_model_path must be non-empty.")

        self._log(
            "loading prompt model "
            f"model_path={self.model_path!r} device_map={self.device_map!r} "
            f"torch_dtype={self.torch_dtype_name!r} "
            f"trust_remote_code={self.trust_remote_code} provider={self.provider}"
        )
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                f"--{self.mode}_qwen_prompt requires transformers with "
                "AutoModelForImageTextToText and AutoProcessor. Install or "
                "upgrade transformers in this environment."
            ) from exc

        load_kwargs = self._model_load_kwargs()

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        if self._uses_explicit_cpu() or self._uses_transient_cuda():
            _qwen_model_to_device(self.model, torch.device("cpu"))
        self.model.eval()
        self._log("prompt model loaded")

    def _timing_scope(self, timing_reporter, name, **metadata):
        reporter = timing_reporter or NOOP_TIMING_REPORTER
        return reporter.scope(f"{self.mode}.qwen_prompt.{name}", metadata=metadata)

    def _instruction(
        self,
        image_count,
        *,
        prompt_mode="oracle_section_gt",
        event=None,
        camera_summary="",
        dataset_prompt="",
        section_idx=0,
        total_sections=None,
        image_source="",
    ):
        if self.mode == "train":
            word_limit = max(1, int(self.word_limit or 200))
            return (
                "Describe the input video as one coherent English video-generation "
                f"prompt around {word_limit} words, including scene and objects. "
                "Treat the input images as temporally ordered frames from one "
                "continuous video segment."
                "Infer a coherent scene and object dynamics across time "
                "Most words should describe the objects in the video in detail, "
                "including their shape, color, texture, material, readable text if "
                "visible, and approximate location in the frame. Use only a few "
                "words for atmosphere, lighting, and other overall information. "
                "Avoid starting with phrases such as 'in the picture' or 'this is "
                "a picture'. Do not describe the frames one by one, enumerate frame "
                "numbers, or write phrases such as first frame, second frame, or "
                "third frame. Do not include information related to the camera. Do "
                "not include reasoning, bullet points, markdown, labels, or line "
                "breaks. Return only the final prompt in one paragraph."
            )
        return _qwen_video_prompt_instruction(
            image_count,
            word_limit=self.word_limit,
            prompt_mode=prompt_mode,
            event=event,
            camera_summary=camera_summary,
            dataset_prompt=dataset_prompt,
            section_idx=section_idx,
            total_sections=total_sections,
            image_source=image_source,
        )

    def _build_inputs(
        self,
        images,
        *,
        prompt_mode="oracle_section_gt",
        event=None,
        camera_summary="",
        dataset_prompt="",
        section_idx=0,
        total_sections=None,
        image_source="",
    ):
        if isinstance(images, Image.Image):
            image_list = [images]
        else:
            image_list = list(images or [])
        if not image_list:
            raise ValueError("Qwen prompt generation requires at least one image.")
        instruction = self._instruction(
            len(image_list),
            prompt_mode=prompt_mode,
            event=event,
            camera_summary=camera_summary,
            dataset_prompt=dataset_prompt,
            section_idx=section_idx,
            total_sections=total_sections,
            image_source=image_source,
        )
        content = [{"type": "image", "image": image} for image in image_list]
        content.append({"type": "text", "text": instruction})
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        chat_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": False,
        }
        try:
            inputs = self.processor.apply_chat_template(messages, **chat_kwargs)
            return inputs, "apply_chat_template(tokenize=True, enable_thinking=False)"
        except TypeError as exc:
            self._log(
                "WARNING: processor.apply_chat_template rejected tokenized "
                f"non-thinking inputs ({exc}); falling back."
            )
            chat_kwargs.pop("enable_thinking", None)
            try:
                inputs = self.processor.apply_chat_template(messages, **chat_kwargs)
                return inputs, "apply_chat_template(tokenize=True)"
            except TypeError:
                pass

        text_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            text = self.processor.apply_chat_template(messages, **text_kwargs)
        except TypeError as exc:
            self._log(
                "WARNING: processor.apply_chat_template rejected "
                f"enable_thinking=False in text mode ({exc}); retrying without it."
            )
            text_kwargs.pop("enable_thinking", None)
            text = self.processor.apply_chat_template(messages, **text_kwargs)
        inputs = self.processor(
            text=[text],
            images=image_list,
            return_tensors="pt",
        )
        return inputs, "processor(text=[...], images=[...])"

    @torch.no_grad()
    def generate_prompt(
        self,
        images,
        *,
        batch_idx,
        section_idx,
        clean_name,
        dataset_prompt,
        timing_reporter=None,
        transient_cuda_device=None,
        prompt_mode="oracle_section_gt",
        event=None,
        camera_summary="",
        total_sections=None,
        image_source="",
    ):
        with self._timing_scope(
            timing_reporter,
            "load_model",
            model_loaded=bool(self.model is not None and self.processor is not None),
        ):
            self._load_once()
        image_list = [images] if isinstance(images, Image.Image) else list(images or [])
        image_sizes = [getattr(image, "size", None) for image in image_list]
        dataset_prompt = dataset_prompt or ""
        self._log(
            f"batch={batch_idx} section={section_idx} clean_name={clean_name!r} "
            f"image_sizes={image_sizes} dataset_prompt_len={len(dataset_prompt)} "
            f"word_limit={self.word_limit} max_new_tokens={self.max_new_tokens} "
            f"mode={prompt_mode} source={image_source} camera={camera_summary!r} "
            f"event={event!r}"
        )

        with self._timing_scope(
            timing_reporter,
            "build_inputs",
            image_count=len(image_list),
            image_sizes=[tuple(int(v) for v in size) for size in image_sizes if size],
            dataset_prompt_len=len(dataset_prompt),
            word_limit=int(self.word_limit),
            event=str(event or ""),
            prompt_mode=str(prompt_mode),
            camera_summary=str(camera_summary or ""),
            image_source=str(image_source or ""),
        ):
            inputs, input_mode = self._build_inputs(
                image_list,
                prompt_mode=prompt_mode,
                event=event,
                camera_summary=camera_summary,
                dataset_prompt=dataset_prompt,
                section_idx=section_idx,
                total_sections=total_sections,
                image_source=image_source,
            )
        transient_device = None
        if self._uses_transient_cuda():
            with self._timing_scope(timing_reporter, "onload_transient_cuda"):
                transient_device = self._transient_cuda_target_device(
                    transient_cuda_device
                )
                self._log(f"onloading prompt model to {transient_device}")
                _qwen_model_to_device(self.model, transient_device)
        try:
            with self._timing_scope(timing_reporter, "resolve_device"):
                device = transient_device or _qwen_model_input_device(self.model)
            with self._timing_scope(
                timing_reporter,
                "move_inputs_to_device",
                device=str(device),
                input_mode=input_mode,
            ):
                inputs = _move_qwen_inputs_to_device(inputs, device)
            with self._timing_scope(timing_reporter, "inspect_inputs"):
                input_ids = _qwen_inputs_input_ids(inputs)
            input_len = int(input_ids.shape[-1]) if input_ids is not None else None
            self._log(
                f"inputs ready mode={input_mode} device={device} input_len={input_len}"
            )

            with self._timing_scope(
                timing_reporter,
                "model_generate",
                input_len=input_len,
                max_new_tokens=int(self.max_new_tokens),
                do_sample=False,
            ):
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            with self._timing_scope(timing_reporter, "trim_generated_ids"):
                if input_ids is not None:
                    try:
                        generated_ids = [
                            output_ids[len(input_ids_item) :]
                            for input_ids_item, output_ids in zip(
                                input_ids, generated_ids
                            )
                        ]
                    except Exception:
                        generated_ids = generated_ids[:, input_ids.shape[-1] :]
            output_len = None
            try:
                output_len = int(generated_ids[0].shape[-1])
            except Exception:
                pass
            with self._timing_scope(
                timing_reporter,
                "batch_decode",
                output_len=output_len,
            ):
                decoded = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            raw_prompt = decoded[0] if decoded else ""
            raw_preview = re.sub(r"\s+", " ", str(raw_prompt)).strip()
            self._log(f"raw prompt len={len(raw_prompt)}: {raw_preview!r}")

            with self._timing_scope(
                timing_reporter,
                "sanitize_prompt",
                raw_prompt_len=len(raw_prompt),
                word_limit=int(self.word_limit),
            ):
                prompt = _sanitize_validation_qwen_prompt(
                    raw_prompt,
                    word_limit=self.word_limit,
                )
            if not prompt:
                raise RuntimeError(
                    "[validation][qwen] Qwen returned an empty prompt after cleanup."
                )
            word_count = len(prompt.split())
            self._log(f"final prompt words={word_count}: {prompt!r}")
            return prompt
        finally:
            inputs = None
            input_ids = None
            generated_ids = None
            decoded = None
            if self._uses_transient_cuda():
                with self._timing_scope(timing_reporter, "offload_transient_cuda"):
                    self._log("offloading prompt model to cpu")
                    _qwen_model_to_device(self.model, torch.device("cpu"))
                    _release_cached_memory()


class _ValidationQwenPromptGenerator(_QwenPromptGenerator):
    def __init__(self, args, rank=0):
        super().__init__(args, mode="validation", rank=rank)


class _TrainQwenPromptGenerator(_QwenPromptGenerator):
    def __init__(self, args, rank=0):
        super().__init__(args, mode="train", rank=rank)
