from .common import *
from .config import TRAIN_PROMPT_DROPOUT_PREVIEW_TEXT

def _prompt_context_from_data(data):
    if not isinstance(data, dict):
        return ""
    info = data.get("info") or {}
    clean_name = info.get("clean_name")
    if clean_name:
        return f" clean_name={clean_name}"
    return ""


def _require_nonempty_prompt(
    prompt,
    *,
    phase,
    data=None,
    clean_name=None,
    allow_empty=False,
):
    text = str(prompt or "").strip()
    if text:
        return text
    context = ""
    if clean_name is not None:
        context = f" clean_name={clean_name}"
    elif data is not None:
        context = _prompt_context_from_data(data)
    if allow_empty:
        key = (str(phase), str(context))
        warned = getattr(_require_nonempty_prompt, "_warned_empty", set())
        if key not in warned and len(warned) < 32:
            warned.add(key)
            setattr(_require_nonempty_prompt, "_warned_empty", warned)
            print(
                f"[prompt] WARNING: {phase} received empty prompt{context}; "
                "continuing because empty prompts are explicitly allowed.",
                flush=True,
            )
        return ""
    raise ValueError(f"{phase} received empty prompt{context}.")


def _normalize_train_prompt_dropout_prob(probability):
    if probability is None:
        probability = 0.0
    probability = float(probability)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(
            "train_prompt_dropout_prob must be in [0, 1], " f"got {probability}."
        )
    return probability


def _is_train_prompt_dropout_sample(data):
    if not isinstance(data, dict):
        return False
    debug = data.get("train_prompt_dropout_debug") or {}
    return bool(debug.get("enabled") and debug.get("dropped"))


def _maybe_apply_train_prompt_dropout(data, *, probability, rng=None):
    probability = _normalize_train_prompt_dropout_prob(probability)
    if not isinstance(data, dict):
        return data
    original_prompt = str(data.get("prompt") or "")
    dropped = probability > 0.0 and (rng or random.random)() < probability
    debug = {
        "enabled": bool(probability > 0.0),
        "dropped": bool(dropped),
        "probability": float(probability),
        "original_prompt": original_prompt,
        "original_prompt_len": int(len(original_prompt)),
    }
    data["train_prompt_dropout_debug"] = debug
    if dropped:
        data["prompt"] = ""
    return data


def _train_prompt_from_data(data, *, phase="train", allow_empty=False):
    prompt = data.get("prompt") if isinstance(data, dict) else None
    if _is_train_prompt_dropout_sample(data):
        return str(prompt or "")
    return _require_nonempty_prompt(
        prompt,
        phase=phase,
        data=data,
        allow_empty=allow_empty,
    )


def _prompt_text_for_preview(data):
    if _is_train_prompt_dropout_sample(data):
        return TRAIN_PROMPT_DROPOUT_PREVIEW_TEXT
    if not isinstance(data, dict):
        return ""
    return data.get("prompt") or ""


# ---------------------------------------------------------------------------
# Training module with mosaic support
# ---------------------------------------------------------------------------
