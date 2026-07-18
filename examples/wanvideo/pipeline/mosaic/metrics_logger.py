"""Training metrics sink: wandb (online/offline) -> tensorboard -> no-op.

The backend is probed ONCE at trainer startup, on the main process only:

* ``wandb_mode=auto`` (default): use remote wandb only when BOTH a usable
  login (env ``WANDB_API_KEY`` or ``~/.netrc``) and connectivity to the
  wandb API host are present; otherwise degrade to ``wandb offline`` with
  the run stored under ``<log_dir>/wandb`` (upload later via
  ``wandb sync <log_dir>/wandb/offline-run-*``).
* When the wandb package is missing (or ``wandb.init`` keeps failing),
  scalars go to tensorboard under ``<log_dir>/tensorboard``.
* If tensorboard is unavailable too, logging becomes a silent no-op --
  telemetry must never block or crash training.

Besides the regular scalars (loss / lr / epoch / global_step / nonfinite
flag), each step is bucketed by the actually-trained clean-history plan
mode so the C / A / B loss curves can be compared directly:

* ``train/loss_mode_single``     -- single-frame anchor, no context.
* ``train/loss_mode_block``      -- block anchor (joint encode), no context.
* ``train/loss_mode_single_ctx`` -- single anchor + discrete context blocks
  (the inference pool-history contract).
* ``train/loss_mode_block_ctx``  -- block anchor + discrete context blocks.
* ``train/loss_mode_off``        -- plan disabled (eval-only batches).
(Old runs logged C / A / B letters; the v2 sampler writes ``bucket``.)
"""

from .common import *  # noqa: F401,F403 -- os/json/time/contextlib/torch et al.


_WANDB_API_HOST = "api.wandb.ai"


def _wandb_logged_in(wandb_module):
    """True when a usable API key is configured (env var or ~/.netrc)."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return bool(wandb_module.api.api_key)
    except Exception:
        return False


def _wandb_network_ok(host=_WANDB_API_HOST, port=443, timeout_s=3.0):
    """Cheap TCP reachability probe of the wandb API host."""
    import socket

    try:
        with contextlib.closing(
            socket.create_connection((host, port), timeout=timeout_s)
        ):
            return True
    except OSError:
        return False


def _decide_wandb_mode(requested, *, logged_in, network_ok):
    """Resolve the effective wandb mode from the CLI request + probes."""
    requested = str(requested or "auto").strip().lower()
    if requested in ("online", "offline", "disabled"):
        return requested
    return "online" if (logged_in and network_ok) else "offline"


class MosaicMetricsLogger:
    """Rank-0 metrics logger with automatic backend selection.

    Construct it on every rank (non-main ranks become no-ops) right after
    ``accelerator.prepare``; call :meth:`log_train_step` once per batch and
    :meth:`finish` at the end of training. All log calls are exception
    safe: a broken backend degrades to a one-time warning, never a crash.
    """

    def __init__(self, args, log_dir, is_main_process):
        self.backend = "noop"
        self._wandb = None
        self._wandb_run = None
        self._tb_writer = None
        self._batch_counter = 0
        self._log_failure_warned = False
        self._local_log_failure_warned = False
        self._log_dir = str(log_dir)
        self._local_metrics_path = (
            os.path.join(self._log_dir, "loss_metrics.jsonl")
            if is_main_process
            else None
        )
        if not is_main_process:
            return
        requested = str(getattr(args, "wandb_mode", "auto") or "auto").lower()
        if requested != "disabled":
            self._init_wandb(args, requested)
        if self._wandb_run is None:
            self._init_tensorboard()
        print(f"[metrics] backend={self.backend}", flush=True)

    # ------------------------------------------------------------------
    # backend init
    # ------------------------------------------------------------------

    def _init_wandb(self, args, requested):
        try:
            import wandb
        except ImportError:
            print(
                "[metrics] wandb package not installed; "
                "falling back to tensorboard.",
                flush=True,
            )
            return
        logged_in = _wandb_logged_in(wandb)
        # Only probe the network when a login exists -- without credentials
        # the run is offline regardless, and the probe just wastes startup
        # time on air-gapped clusters.
        network_ok = _wandb_network_ok() if logged_in else False
        mode = _decide_wandb_mode(requested, logged_in=logged_in, network_ok=network_ok)
        if mode == "disabled":
            return
        if mode == "offline" and requested == "auto":
            print(
                "[metrics] wandb status check: "
                f"logged_in={logged_in} network_ok={network_ok} -> "
                "running OFFLINE; run data is stored under "
                f"{os.path.join(self._log_dir, 'wandb')} and can be uploaded "
                "later with `wandb sync`.",
                flush=True,
            )

        try:
            from .config import _run_args_to_yaml_payload

            config_payload = _run_args_to_yaml_payload(args)
        except Exception:
            config_payload = None
        run_name = str(getattr(args, "wandb_run_name", "") or "") or (
            os.path.basename(os.path.normpath(self._log_dir))
        )
        init_kwargs = dict(
            project=str(getattr(args, "wandb_project", "") or "") or "diffsynth-mosaic",
            name=run_name,
            dir=self._log_dir,
            mode=mode,
            config=config_payload,
        )
        entity = str(getattr(args, "wandb_entity", "") or "")
        if entity:
            init_kwargs["entity"] = entity

        os.makedirs(self._log_dir, exist_ok=True)
        try:
            self._wandb_run = wandb.init(**init_kwargs)
        except Exception as exc:
            if mode == "online":
                # Connectivity can vanish between the probe and init (or
                # the key may be stale); retry locally instead of dying.
                print(
                    f"[metrics] wandb.init(online) failed ({exc!r}); "
                    "retrying in offline mode.",
                    flush=True,
                )
                init_kwargs["mode"] = mode = "offline"
                try:
                    self._wandb_run = wandb.init(**init_kwargs)
                except Exception as exc2:
                    print(
                        f"[metrics] wandb.init(offline) failed too ({exc2!r}); "
                        "falling back to tensorboard.",
                        flush=True,
                    )
                    return
            else:
                print(
                    f"[metrics] wandb.init({mode}) failed ({exc!r}); "
                    "falling back to tensorboard.",
                    flush=True,
                )
                return
        self._wandb = wandb
        self.backend = f"wandb-{mode}"

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print(
                "[metrics] tensorboard not available either; "
                "metrics logging disabled.",
                flush=True,
            )
            return
        tb_dir = os.path.join(self._log_dir, "tensorboard")
        try:
            os.makedirs(tb_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=tb_dir)
        except Exception as exc:
            print(
                f"[metrics] tensorboard writer init failed ({exc!r}); "
                "metrics logging disabled.",
                flush=True,
            )
            self._tb_writer = None
            return
        self.backend = "tensorboard"

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------

    def log_scalars(self, scalars, step=None):
        """Log a flat ``{name: number}`` dict at ``step`` (defaults to the
        internal batch counter). No-op on non-main ranks / dead backends."""
        if not scalars:
            return
        if self.backend == "noop":
            return
        if step is None:
            step = self._batch_counter
        try:
            if self._wandb_run is not None:
                self._wandb_run.log(
                    {str(k): v for k, v in scalars.items()}, step=int(step)
                )
            elif self._tb_writer is not None:
                for key, value in scalars.items():
                    self._tb_writer.add_scalar(str(key), float(value), int(step))
        except Exception as exc:
            if not self._log_failure_warned:
                self._log_failure_warned = True
                print(
                    f"[metrics] logging failed once ({exc!r}); further "
                    "failures are silenced so training is not disturbed.",
                    flush=True,
                )

    def _write_local_metrics(self, scalars, step=None):
        """Append per-step metrics to a local JSONL file on rank 0.

        This is independent of wandb/tensorboard so k8s runs always leave an
        easy-to-download scalar trace under the run log dir.
        """
        if not scalars or not self._local_metrics_path:
            return
        if step is None:
            step = self._batch_counter
        payload = {"step": int(step), "time": time.time()}
        for key, value in scalars.items():
            try:
                payload[str(key)] = float(value)
            except (TypeError, ValueError):
                pass
        try:
            with open(self._local_metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception as exc:
            if not self._local_log_failure_warned:
                self._local_log_failure_warned = True
                print(
                    f"[metrics] local JSONL write failed once ({exc!r}); "
                    "further failures are silenced so training is not disturbed.",
                    flush=True,
                )

    def log_train_step(
        self,
        *,
        loss,
        lr,
        epoch,
        global_step,
        data=None,
        svi_metrics=None,
        nonfinite=False,
        loss_local=None,
    ):
        """Per-batch entry point (call on every rank; non-main is a no-op).

        ``loss`` is the headline scalar (caller may pass a cross-rank mean so
        ``train/loss`` reflects the whole global batch). ``loss_local`` is the
        rank0-local loss used for the per-mode buckets / anchor flags, which
        are attributed to *this* rank's sampled plan and so must not use a
        cross-rank average; it defaults to ``loss`` when not supplied.

        ``data`` is the training batch dict AFTER the forward pass --
        ``_apply_train_clean_latent_history`` leaves the actually-sampled
        plan in ``data['train_clean_latent_history_debug']``, which feeds
        the per-mode loss buckets. Cached-latent batches without the debug
        payload simply skip the bucketed scalars.
        """
        self._batch_counter += 1
        if self.backend == "noop":
            return
        bucket_loss = float(loss if loss_local is None else loss_local)
        scalars = {
            "train/loss": float(loss),
            "train/lr": float(lr),
            "train/epoch": float(epoch),
            "train/global_step": float(global_step),
            "train/nonfinite": 1.0 if nonfinite else 0.0,
        }
        debug = None
        if isinstance(data, dict):
            maybe = data.get("train_clean_latent_history_debug")
            if isinstance(maybe, dict):
                debug = maybe
            loss_metrics = data.get("loss_metrics")
            if isinstance(loss_metrics, dict):
                for key, value in loss_metrics.items():
                    try:
                        scalars[f"train/{key}"] = float(value)
                    except (TypeError, ValueError):
                        pass
        if debug is not None:
            mode = self._plan_mode(debug)
            if mode is not None:
                scalars[f"train/loss_mode_{mode}"] = bucket_loss
            if debug.get("enabled"):
                # v2 plans carry context_count; legacy debug dicts carried
                # history_count (= context blocks incl. the anchor slot).
                context_count = debug.get("context_count")
                history_count = (
                    int(context_count) + 1
                    if context_count is not None
                    else debug.get("history_count")
                )
                if history_count is not None:
                    scalars["train/history_count"] = float(history_count)
            anchor_form = debug.get("anchor_form")
            if anchor_form:
                scalars["train/anchor_is_single"] = (
                    1.0 if str(anchor_form) == "single" else 0.0
                )
        if isinstance(svi_metrics, dict):
            for key, value in svi_metrics.items():
                if isinstance(value, (int, float)):
                    scalars[f"svi/{key}"] = float(value)
        self._write_local_metrics(scalars, step=self._batch_counter)
        self.log_scalars(scalars, step=self._batch_counter)

    @staticmethod
    def _plan_mode(debug):
        """Map the plan debug dict to a loss-bucket suffix.

        v2 plans carry an explicit ``bucket`` label
        (single / block / single_ctx / block_ctx); legacy debug dicts
        fall back to the old C/A/B letters. Disabled plans -> "off".
        """
        if not debug.get("enabled"):
            return "off"
        bucket = debug.get("bucket")
        if bucket:
            return str(bucket)
        mode = debug.get("mode")
        return str(mode) if mode else None

    def log_validation_done(self, epoch_id, global_step):
        """Mark a finished validation pass on the metrics timeline."""
        self.log_scalars(
            {
                "val/completed_epoch": float(epoch_id),
                "val/at_global_step": float(global_step),
            }
        )

    # ------------------------------------------------------------------
    # teardown
    # ------------------------------------------------------------------

    def finish(self):
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
            self._wandb_run = None
        if self._tb_writer is not None:
            try:
                self._tb_writer.flush()
                self._tb_writer.close()
            except Exception:
                pass
            self._tb_writer = None
