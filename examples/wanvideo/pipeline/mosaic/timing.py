from .common import *
from .common import _glob

_TIMING_NODE_ENV_KEYS = (
    "NODE_RANK",
    "GROUP_RANK",
    "MACHINE_RANK",
    "SLURM_NODEID",
    "OMPI_COMM_WORLD_NODE_RANK",
    "MV2_COMM_WORLD_NODE_RANK",
)
_TIMING_WRITER_SPAWN_NS = time.time_ns()


def _dump_path(path):
    if not os.path.exists(path):
        return
    dump_path = f"{path}.dump"
    try:
        os.replace(path, dump_path)
    except OSError:
        pass


class _NoopTimingReporter:
    enabled = False

    @contextlib.contextmanager
    def record(self, record_type, **metadata):
        yield self

    @contextlib.contextmanager
    def scope(self, name, metadata=None):
        yield self

    def add_duration(self, name, duration_s, metadata=None):
        return None

    def set_metadata(self, **kwargs):
        return None

    def start_record(self, record_type, **metadata):
        return False

    def finish_record(self):
        return None

    def consume_postfix(self):
        return {}

    def write_cache_diagnostic(self, record):
        return None


NOOP_TIMING_REPORTER = _NoopTimingReporter()


RECENT_FIRST_ZBUFFER_MODE = "recent_first_zbuffer"
MOSAIC_INTRINSICS_MODES = ("per_frame", "episode_mean", "first_frame")
TRAIN_PROMPT_DROPOUT_PREVIEW_TEXT = "<no_prompt>"


def _sanitize_timing_tag(value):
    text = str(value or "").strip()
    if not text:
        return None
    sanitized = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text
    ).strip("_")
    return sanitized or None


def _timing_writer_tag():
    for key in _TIMING_NODE_ENV_KEYS:
        value = _sanitize_timing_tag(os.environ.get(key))
        if value is None:
            continue
        if value.isdigit():
            return f"node{int(value):03d}"
        return f"node{value}"
    return f"spawn{_TIMING_WRITER_SPAWN_NS}"


class TimingReporter:
    """Lightweight JSON timing reporter for distributed training ranks."""

    def __init__(
        self,
        log_dir,
        *,
        rank=0,
        is_main_process=False,
        enabled=True,
        interval=20,
        warmup_steps=1,
        dirname="timing",
        cuda_sync=True,
        print_top_k=5,
        window_size=50,
    ):
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.is_main_process = bool(is_main_process)
        self.interval = max(1, int(interval))
        self.warmup_steps = max(0, int(warmup_steps))
        self.cuda_sync = bool(cuda_sync)
        self.print_top_k = max(1, int(print_top_k))
        self.window_size = max(1, int(window_size))
        self._active_record = None
        self._active_record_start = None
        self._train_records_seen = 0
        self._rolling = defaultdict(lambda: deque(maxlen=self.window_size))
        self._last_postfix = {}
        self.writer_tag = _timing_writer_tag()
        self.output_dir = os.path.join(log_dir, dirname)
        timing_stem = f"timing_{self.writer_tag}_rank{self.rank:03d}"
        self.json_path = os.path.join(self.output_dir, f"{timing_stem}.json")
        self.cache_diagnostic_path = os.path.join(
            self.output_dir,
            f"cache_diagnostics_{self.writer_tag}_rank{self.rank:03d}.json",
        )
        self.gantt_path = os.path.join(self.output_dir, f"{timing_stem}_latest.html")
        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)

    def _sync_cuda(self):
        if not self.cuda_sync:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (AssertionError, RuntimeError):
            return

    def _now(self, *, sync=True):
        if sync:
            self._sync_cuda()
        return time.perf_counter()

    @contextlib.contextmanager
    def record(self, record_type, **metadata):
        started = self.start_record(record_type, **metadata)
        if not started:
            yield self
            return
        try:
            yield self
        finally:
            self.finish_record()

    def start_record(self, record_type, **metadata):
        if not self.enabled:
            return False
        if self._active_record is not None:
            return False
        start = self._now()
        self._active_record_start = start
        self._active_record = {
            "type": record_type,
            "rank": self.rank,
            "writer": self.writer_tag,
            "cuda_sync": self.cuda_sync,
            "metadata": dict(metadata),
            "events": [],
        }
        return True

    def finish_record(self):
        if not self.enabled or self._active_record is None:
            return None
        end = self._now()
        record = self._active_record
        start = self._active_record_start
        self._active_record = None
        self._active_record_start = None
        record["duration_s"] = max(0.0, end - start)
        record["event_totals_s"] = self._event_totals(record["events"])
        record["cuda_memory"] = self._cuda_memory()
        self._maybe_write_record(record)
        return record

    @contextlib.contextmanager
    def scope(self, name, metadata=None):
        if not self.enabled or self._active_record is None:
            yield self
            return
        start = self._now()
        try:
            yield self
        finally:
            end = self._now()
            self._active_record["events"].append(
                {
                    "name": str(name),
                    "start_ms": round((start - self._active_record_start) * 1000.0, 3),
                    "duration_s": max(0.0, end - start),
                    "metadata": dict(metadata or {}),
                }
            )

    def add_duration(self, name, duration_s, metadata=None):
        if not self.enabled or self._active_record is None:
            return None
        self._active_record["events"].append(
            {
                "name": str(name),
                "start_ms": 0.0,
                "duration_s": max(0.0, float(duration_s)),
                "metadata": dict(metadata or {}),
            }
        )
        return None

    def set_metadata(self, **kwargs):
        if self.enabled and self._active_record is not None:
            self._active_record["metadata"].update(kwargs)

    def consume_postfix(self):
        postfix = dict(self._last_postfix)
        self._last_postfix = {}
        return postfix

    def _maybe_write_record(self, record):
        if record["type"] == "train_step":
            self._train_records_seen += 1
            if self._train_records_seen <= self.warmup_steps:
                return
            self._update_rolling(record)
            self._last_postfix = self._format_postfix(record)
        self._write_json(record)
        if record["type"] == "train_step":
            self._write_latest_gantt(record)
            if self.is_main_process and self._train_records_seen % self.interval == 0:
                print(self._format_console_summary(record))

    def _update_rolling(self, record):
        self._rolling["total"].append(float(record["duration_s"]))
        for name, duration in record["event_totals_s"].items():
            self._rolling[name].append(float(duration))

    def _format_postfix(self, record):
        postfix = {"time": f"{self._mean(self._rolling['total']):.2f}s"}
        top = self._top_events(record["event_totals_s"], limit=2)
        if top:
            postfix["slow"] = ",".join(
                f"{name.split('.')[-1]}={value:.2f}s" for name, value in top
            )
        return postfix

    def _format_console_summary(self, record):
        top = self._top_events(
            {
                name: self._mean(values)
                for name, values in self._rolling.items()
                if name != "total"
            },
            limit=self.print_top_k,
        )
        top_text = ", ".join(f"{name}={value:.3f}s" for name, value in top) or "none"
        step = record["metadata"].get("global_step", "?")
        total = self._mean(self._rolling["total"])
        return f"[timing][rank {self.rank}] step={step} avg_total={total:.3f}s top={top_text}"

    def _write_json(self, record):
        self._append_json_record(self.json_path, record)

    @staticmethod
    def _append_json_record(path, record):
        records = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    records = payload
            except (OSError, ValueError):
                records = []
        records.append(record)
        tmp_path = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, sort_keys=True)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    _dump_path(tmp_path)
                except OSError:
                    pass

    def write_cache_diagnostic(self, record):
        if not self.enabled:
            return None
        payload = dict(record or {})
        payload.setdefault("rank", self.rank)
        payload.setdefault("writer", self.writer_tag)
        payload.setdefault("time_s", time.time())
        if self._active_record is not None:
            payload.setdefault("active_record_type", self._active_record.get("type"))
            payload.setdefault(
                "active_record_metadata",
                dict(self._active_record.get("metadata") or {}),
            )
        parent = os.path.dirname(self.cache_diagnostic_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._append_json_record(self.cache_diagnostic_path, payload)
        return self.cache_diagnostic_path

    @staticmethod
    def _format_int(value, default="n/a"):
        if value is None:
            return default
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return html.escape(str(value))

    @staticmethod
    def _format_pct(value, default="n/a"):
        if value is None:
            return default
        try:
            return f"{float(value) * 100.0:.1f}%"
        except (TypeError, ValueError):
            return html.escape(str(value))

    def _format_mosaic_sequence_section(self, stats):
        if not isinstance(stats, dict) or not stats:
            return ""

        status = html.escape(str(stats.get("status", "unknown")))
        drop_enabled = bool(stats.get("drop_holes_enabled", False))
        before = stats.get("seq_len_before_drop")
        after = stats.get("seq_len_after_drop")
        dropped = stats.get("dropped_tokens")
        keep_ratio = stats.get("keep_ratio")
        drop_ratio = stats.get("drop_ratio")

        def _as_int(value):
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0

        reference_tokens = _as_int(stats.get("reference_tokens"))
        first_tokens = _as_int(stats.get("first_tokens"))
        noisy_tokens = _as_int(stats.get("noisy_tokens"))
        mosaic_kept = _as_int(stats.get("mosaic_tokens_after"))
        mosaic_hole = _as_int(stats.get("mosaic_hole_tokens"))
        dropped_tokens = _as_int(dropped)

        # Bar segments are (label, css-kind, count, in_sequence). The bar width
        # always represents the PRE-drop sequence length so the two hole modes
        # read correctly:
        #   - drop mode: hole tokens are physically removed, so they render as a
        #     trailing band flagged in_sequence=False (the live sequence visibly
        #     shrinks and leaves a hatched "removed" gap).
        #   - mask mode: hole tokens stay in the sequence (just zeroed), so they
        #     render as an in-sequence "masked" band. There is NO "dropped" band
        #     because nothing was actually dropped.
        segments = [
            ("reference", "reference", reference_tokens, True),
            ("first", "first", first_tokens, True),
            ("mosaic kept", "mosaic-kept", mosaic_kept, True),
        ]
        if drop_enabled:
            segments.append(
                ("mosaic dropped (removed)", "mosaic-dropped", dropped_tokens, False)
            )
        else:
            segments.append(
                ("mosaic holes (masked)", "mosaic-holes-masked", mosaic_hole, True)
            )
        segments.append(("noisy", "noisy", noisy_tokens, True))

        bar_total = _as_int(before)
        if bar_total <= 0:
            bar_total = sum(count for _, _, count, _ in segments)
        segment_html = []
        for label, kind, count, in_sequence in segments:
            if count <= 0 or bar_total <= 0:
                continue
            width = max(0.6, count / bar_total * 100.0)
            seq_attr = "1" if in_sequence else "0"
            segment_html.append(
                "<div class='seq-segment' "
                f"data-kind='{kind}' data-in-seq='{seq_attr}' "
                f"style='width:{width:.3f}%'>"
                f"<span>{html.escape(label)} {count:,}</span>"
                "</div>"
            )
        bar = (
            "\n".join(segment_html) or "<div class='seq-empty'>no sequence stats</div>"
        )
        note = (
            "holes are physically removed before DiT attention "
            "(hatched red band = removed from sequence)"
            if drop_enabled
            else "holes are masked/zeroed but still occupy sequence positions "
            "(no length change)"
        )

        return (
            "<section class='mosaic-seq'>"
            "<h2>Mosaic Sequence Length</h2>"
            "<div class='seq-cards'>"
            f"<div class='seq-card'><span>status</span><strong>{status}</strong></div>"
            f"<div class='seq-card'><span>seq len</span><strong>{self._format_int(before)} &rarr; {self._format_int(after)}</strong></div>"
            f"<div class='seq-card'><span>dropped</span><strong>{self._format_int(dropped)}</strong></div>"
            f"<div class='seq-card'><span>keep ratio</span><strong>{self._format_pct(keep_ratio)}</strong></div>"
            f"<div class='seq-card'><span>drop ratio</span><strong>{self._format_pct(drop_ratio)}</strong></div>"
            "</div>"
            f"<div class='seq-bar'>{bar}</div>"
            "<div class='seq-details'>"
            f"tokens/frame={self._format_int(stats.get('tokens_per_frame'))}; "
            f"mosaic_frames={self._format_int(stats.get('mosaic_frame_count'))}; "
            f"noisy_frames={self._format_int(stats.get('noisy_frame_count'))}; "
            f"mosaic_holes={self._format_int(mosaic_hole)}; "
            f"cross_attn_masked={self._format_int(stats.get('cross_attn_masked_tokens'))}; "
            f"{html.escape(note)}"
            "</div>"
            "</section>"
        )

    def _format_train_qwen_prompt_section(self, info):
        if not isinstance(info, dict) or not info.get("enabled"):
            return ""
        sampled = info.get("sampled_frame_indices") or []
        sizes = info.get("image_sizes") or []
        prompt = html.escape(str(info.get("qwen_prompt", "")))
        original = html.escape(str(info.get("original_prompt", "")))
        return (
            "<section class='qwen-prompt'>"
            "<h2>Train Qwen Prompt</h2>"
            "<div class='seq-cards'>"
            f"<div class='seq-card'><span>clean</span><strong>{html.escape(str(info.get('clean_name', 'unknown')))}</strong></div>"
            f"<div class='seq-card'><span>frames</span><strong>{len(sampled)}</strong></div>"
            f"<div class='seq-card'><span>words</span><strong>{self._format_int(info.get('qwen_prompt_words'))}</strong></div>"
            f"<div class='seq-card'><span>max side</span><strong>{self._format_int(info.get('frame_max_side'))}</strong></div>"
            "</div>"
            "<div class='qwen-details'>"
            f"model={html.escape(str(info.get('model_path', '')))}<br>"
            f"sampled_frame_indices={html.escape(str(sampled))}<br>"
            f"image_sizes={html.escape(str(sizes))}"
            "</div>"
            "<h3>Qwen prompt</h3>"
            f"<pre class='prompt-text'>{prompt}</pre>"
            "<h3>Original dataset prompt</h3>"
            f"<pre class='prompt-text muted'>{original}</pre>"
            "</section>"
        )

    def _write_latest_gantt(self, record):
        events = record.get("events", [])
        if not events:
            return
        total_ms = max(1.0, float(record.get("duration_s", 0.0)) * 1000.0)
        palette = [
            "#4f7cff",
            "#f59e0b",
            "#10b981",
            "#ef4444",
            "#8b5cf6",
            "#06b6d4",
            "#f97316",
            "#84cc16",
        ]
        group_colors = {}

        def _event_group(name):
            parts = str(name).split(".")
            if len(parts) >= 2 and parts[0] == "dataset":
                return ".".join(parts[:2])
            return parts[0] or "other"

        def _group_color(group):
            if group not in group_colors:
                group_colors[group] = palette[len(group_colors) % len(palette)]
            return group_colors[group]

        def _event_lane(name):
            name = str(name)
            if name == "train.data_wait":
                return "input wait (before step)"
            if name == "train.model_forward":
                return "model forward (inclusive)"
            if name.startswith("forward."):
                return "inside forward"
            if name.startswith("materialize."):
                return "inside materialize"
            if name == "train.backward":
                return "backward"
            if name.startswith("train."):
                return "train step"
            if name.startswith("dataset."):
                return "dataset worker"
            return "other"

        def _format_ms(value):
            value = float(value)
            if abs(value - round(value)) < 1e-6:
                return f"{int(round(value))} ms"
            return f"{value:.1f} ms"

        axis_ticks = []
        for pct in (0, 25, 50, 75, 100):
            tick_ms = total_ms * pct / 100.0
            axis_ticks.append(
                "<div class='tick' "
                f"style='left:{pct:.3f}%'>"
                f"<span>{_format_ms(tick_ms)}</span>"
                "</div>"
            )
        axis = "\n".join(axis_ticks)
        rows = []
        for event in sorted(events, key=lambda item: float(item.get("start_ms", 0.0))):
            raw_name = str(event["name"])
            name = html.escape(raw_name)
            group = _event_group(raw_name)
            group_label = html.escape(group)
            lane_label = html.escape(_event_lane(raw_name))
            group_color = _group_color(group)
            depth = max(0, raw_name.count("."))
            start_pct = min(
                100.0, max(0.0, float(event.get("start_ms", 0.0)) / total_ms * 100.0)
            )
            width_pct = min(
                100.0 - start_pct,
                max(0.2, float(event["duration_s"]) * 1000.0 / total_ms * 100.0),
            )
            start_ms = _format_ms(float(event.get("start_ms", 0.0)))
            end_ms = _format_ms(
                float(event.get("start_ms", 0.0)) + float(event["duration_s"]) * 1000.0
            )
            rows.append(
                f"<div class='row' data-group=\"{group_label}\" "
                f"style='--group-color: {group_color}; --label-indent: {min(depth, 5) * 12}px;'>"
                f"<div class='label'><span class='event-name'>{name}</span>"
                f"<span class='group'>{group_label}</span>"
                f"<span class='lane'>{lane_label}</span></div>"
                "<div class='track'>"
                f"<div class='bar' title='{name}: {start_ms} - {end_ms}' "
                f"style='left:{start_pct:.3f}%;width:{width_pct:.3f}%;'></div>"
                "</div>"
                f"<div class='duration'>{event['duration_s']:.3f}s</div>"
                "</div>"
            )
        step = html.escape(str(record["metadata"].get("global_step", "?")))
        total = float(record.get("duration_s", 0.0))
        body = "\n".join(rows)
        mosaic_sequence_section = self._format_mosaic_sequence_section(
            record.get("metadata", {}).get("mosaic_sequence_stats")
        )
        train_qwen_prompt_section = self._format_train_qwen_prompt_section(
            record.get("metadata", {}).get("train_qwen_prompt")
        )
        dataset_blocks = []
        dataset_dir = os.path.join(self.output_dir, "dataset")
        if os.path.isdir(dataset_dir):
            try:
                dataset_paths = sorted(
                    _glob.glob(os.path.join(dataset_dir, "*.json")),
                    key=os.path.getmtime,
                    reverse=True,
                )[:4]
            except OSError:
                dataset_paths = []
            for dataset_path in dataset_paths:
                try:
                    with open(dataset_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                except (OSError, ValueError):
                    continue
                if not isinstance(payload, list) or not payload:
                    continue
                dataset_record = payload[-1]
                dataset_events = dataset_record.get("events", [])
                if not dataset_events:
                    continue
                dataset_total_ms = max(
                    1.0, float(dataset_record.get("duration_s", 0.0)) * 1000.0
                )
                dataset_axis_ticks = []
                for pct in (0, 25, 50, 75, 100):
                    tick_ms = dataset_total_ms * pct / 100.0
                    dataset_axis_ticks.append(
                        "<div class='tick' "
                        f"style='left:{pct:.3f}%'>"
                        f"<span>{_format_ms(tick_ms)}</span>"
                        "</div>"
                    )
                dataset_rows = []
                for event in sorted(
                    dataset_events,
                    key=lambda item: float(item.get("start_ms", 0.0)),
                ):
                    raw_name = str(event["name"])
                    name = html.escape(raw_name)
                    group = _event_group(raw_name)
                    group_label = html.escape(group)
                    lane_label = html.escape(_event_lane(raw_name))
                    group_color = _group_color(group)
                    depth = max(0, raw_name.count(".") - 1)
                    start_ms_value = float(event.get("start_ms", 0.0))
                    start_pct = min(
                        100.0, max(0.0, start_ms_value / dataset_total_ms * 100.0)
                    )
                    width_pct = min(
                        100.0 - start_pct,
                        max(
                            0.2,
                            float(event["duration_s"])
                            * 1000.0
                            / dataset_total_ms
                            * 100.0,
                        ),
                    )
                    start_ms = _format_ms(start_ms_value)
                    end_ms = _format_ms(
                        start_ms_value + float(event["duration_s"]) * 1000.0
                    )
                    dataset_rows.append(
                        f"<div class='row' data-group=\"{group_label}\" "
                        f"style='--group-color: {group_color}; --label-indent: {min(depth, 5) * 12}px;'>"
                        f"<div class='label'><span class='event-name'>{name}</span>"
                        f"<span class='group'>{group_label}</span>"
                        f"<span class='lane'>{lane_label}</span></div>"
                        "<div class='track'>"
                        f"<div class='bar' title='{name}: {start_ms} - {end_ms}' "
                        f"style='left:{start_pct:.3f}%;width:{width_pct:.3f}%;'></div>"
                        "</div>"
                        f"<div class='duration'>{event['duration_s']:.3f}s</div>"
                        "</div>"
                    )
                metadata = dataset_record.get("metadata", {})
                dataset_title = html.escape(
                    str(
                        metadata.get("dataset_cls")
                        or dataset_record.get("type")
                        or "dataset"
                    )
                )
                dataset_index = html.escape(str(metadata.get("index", "?")))
                worker = html.escape(str(dataset_record.get("worker", "?")))
                dataset_blocks.append(
                    "<section class='dataset-record'>"
                    f"<h3>{dataset_title} index={dataset_index} worker={worker}</h3>"
                    "<div class='axis'>"
                    + "\n".join(dataset_axis_ticks)
                    + "</div>"
                    + "\n".join(dataset_rows)
                    + "</section>"
                )
        dataset_section = ""
        if dataset_blocks:
            dataset_section = (
                "<h2>Dataset worker timing</h2>"
                "<p class='timeline-note'>Dataset axes are absolute within each dataset record. "
                "They show the latest records written by dataset workers under this rank timing directory.</p>"
                + "\n".join(dataset_blocks)
            )
        content = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Timing rank {self.rank}</title>
<style>
body {{ font-family: sans-serif; margin: 20px; overflow-x: auto; }}
.timeline-note {{ color: #555; margin-bottom: 8px; }}
.timeline {{ min-width: 1320px; }}
.axis {{ position: relative; height: 30px; margin-left: 532px; margin-right: 102px; border-top: 1px solid #999; }}
.tick {{ position: absolute; top: -1px; height: 8px; border-left: 1px solid #999; }}
.tick span {{ position: absolute; top: 10px; transform: translateX(-50%); font-size: 11px; color: #555; white-space: nowrap; }}
.row {{ display: grid; grid-template-columns: 520px minmax(560px, 1fr) 90px; gap: 12px; margin: 6px 0; align-items: center; border-left: 5px solid var(--group-color); padding-left: 6px; }}
.track {{ height: 18px; background: #eee; position: relative; overflow: hidden; box-shadow: inset 0 0 0 1px #ddd; }}
.bar {{ height: 18px; background: var(--group-color); position: absolute; top: 0; opacity: 0.82; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.18); }}
.label {{ padding-left: var(--label-indent); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.25; overflow-wrap: anywhere; word-break: break-word; }}
.event-name {{ display: inline; }}
.group {{ margin-left: 8px; padding: 1px 5px; border-radius: 8px; background: var(--group-color); color: white; font-size: 11px; }}
.lane {{ margin-left: 6px; padding: 1px 5px; border-radius: 8px; background: #eee; color: #555; font-family: sans-serif; font-size: 11px; white-space: nowrap; }}
.duration {{ text-align: right; font-variant-numeric: tabular-nums; }}
.mosaic-seq {{ margin: 16px 0 18px; padding: 14px; border: 1px solid #ddd; border-radius: 10px; background: #fafafa; max-width: 1320px; }}
.mosaic-seq h2 {{ margin: 0 0 10px; }}
.seq-cards {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }}
.seq-card {{ min-width: 150px; padding: 8px 10px; border: 1px solid #ddd; border-radius: 8px; background: white; }}
.seq-card span {{ display: block; color: #666; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
.seq-card strong {{ display: block; margin-top: 3px; font-size: 18px; }}
.seq-bar {{ display: flex; height: 30px; overflow: hidden; border-radius: 8px; background: #eee; box-shadow: inset 0 0 0 1px #ddd; }}
.seq-segment {{ display: flex; align-items: center; justify-content: center; min-width: 28px; color: white; font-size: 11px; white-space: nowrap; overflow: hidden; }}
.seq-segment[data-kind="reference"] {{ background: #64748b; }}
.seq-segment[data-kind="first"] {{ background: #10b981; }}
.seq-segment[data-kind="mosaic-kept"] {{ background: #4f7cff; }}
.seq-segment[data-kind="mosaic-holes-masked"] {{ background: #f59e0b; }}
.seq-segment[data-kind="mosaic-dropped"] {{ background: #ef4444; }}
.seq-segment[data-kind="noisy"] {{ background: #8b5cf6; }}
.seq-segment[data-in-seq="0"] {{ opacity: 0.6; background-image: repeating-linear-gradient(45deg, rgba(255,255,255,0.35) 0 6px, transparent 6px 12px); }}
.seq-details {{ margin-top: 8px; color: #555; font-size: 12px; }}
.qwen-prompt {{ margin: 16px 0 18px; padding: 14px; border: 1px solid #ddd; border-radius: 10px; background: #fbfbfb; max-width: 1320px; }}
.qwen-prompt h2 {{ margin: 0 0 10px; }}
.qwen-prompt h3 {{ margin: 12px 0 6px; font-size: 14px; }}
.qwen-details {{ margin-top: 8px; color: #555; font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; }}
.prompt-text {{ white-space: pre-wrap; overflow-wrap: anywhere; background: white; border: 1px solid #ddd; border-radius: 6px; padding: 8px; margin: 0; font-size: 12px; line-height: 1.4; }}
.prompt-text.muted {{ color: #555; background: #f5f5f5; }}
</style>
</head>
<body>
<h1>Timing rank {self.rank}</h1>
<p>Latest train step: {step}; total: {total:.3f}s; cuda_sync: {self.cuda_sync}</p>
{mosaic_sequence_section}
{train_qwen_prompt_section}
<p class="timeline-note">Absolute time axis: bars are positioned by start_ms from the beginning of this record. Rows are inclusive timing events, not an exclusive flame graph; overlapping rows usually mean nested/inclusive scopes. <code>train.data_wait</code> is input/DataLoader wait before the step, outside forward/backward.</p>
<div class="timeline">
<div class="axis">{axis}</div>
{body}
{dataset_section}
</div>
</body>
</html>
"""
        with open(self.gantt_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _event_totals(self, events):
        totals = defaultdict(float)
        for event in events:
            totals[event["name"]] += float(event["duration_s"])
        return dict(sorted(totals.items()))

    def _cuda_memory(self):
        try:
            if not torch.cuda.is_available():
                return None
            return {
                "allocated": int(torch.cuda.memory_allocated()),
                "reserved": int(torch.cuda.memory_reserved()),
                "max_allocated": int(torch.cuda.max_memory_allocated()),
                "max_reserved": int(torch.cuda.max_memory_reserved()),
            }
        except (AssertionError, RuntimeError, AttributeError):
            return None

    def _top_events(self, totals, limit):
        return sorted(totals.items(), key=lambda item: item[1], reverse=True)[:limit]

    def _mean(self, values):
        values = list(values)
        return sum(values) / len(values) if values else 0.0


def get_timing_reporter(obj):
    return getattr(obj, "timing_reporter", NOOP_TIMING_REPORTER)


def _debug_timing_enabled(args):
    return bool(getattr(args, "debug", False) and getattr(args, "timing_enabled", True))


def _timing_record_event_count(timing_reporter):
    """Number of events in the currently-active record (0 if no record / disabled).

    Used by debug-mode instrumentation to snapshot the inner-scope event count
    before a callee and diff it afterwards, so we can render a breakdown of
    just that callee's contribution without touching reporter internals from
    multiple call sites.
    """
    if timing_reporter is None or not getattr(timing_reporter, "enabled", False):
        return 0
    active = getattr(timing_reporter, "_active_record", None)
    if active is None:
        return 0
    events = active.get("events")
    return len(events) if events else 0


def _format_timing_breakdown(
    timing_reporter,
    baseline_event_count,
    *,
    top_k=15,
    indent="    ",
    exclude_names=None,
):
    """Aggregate (name -> total_s, count) over events appended since baseline.

    Returns a formatted multi-line string. Returns ``""`` when timing is
    disabled or no events were captured (so callers can fall back to a
    simple one-line print).
    """
    if timing_reporter is None or not getattr(timing_reporter, "enabled", False):
        return ""
    active = getattr(timing_reporter, "_active_record", None)
    if active is None:
        return ""
    events = active.get("events") or []
    if baseline_event_count >= len(events):
        return ""
    exclude = set(exclude_names or ())
    agg = defaultdict(lambda: [0.0, 0])
    for ev in events[baseline_event_count:]:
        name = ev.get("name", "")
        if name in exclude:
            continue
        agg[name][0] += float(ev.get("duration_s", 0.0))
        agg[name][1] += 1
    if not agg:
        return ""
    items = sorted(agg.items(), key=lambda kv: -kv[1][0])[: max(1, int(top_k))]
    return "\n".join(
        f"{indent}{name}: {total * 1000.0:.2f}ms ({count}x)"
        for name, (total, count) in items
    )


def _debug_dataset_timing_enabled(args):
    return bool(
        getattr(args, "debug", False) and getattr(args, "dataset_timing_enabled", True)
    )
