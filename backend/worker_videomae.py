from __future__ import annotations

import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import redis
from redis.exceptions import ConnectionError as RedisConnectionError
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from supabase import create_client

from incident_handler import handle_incident

# Load env from backend/.env (preferred) and repo root .env (fallback)
_HERE = os.path.dirname(os.path.realpath(__file__))
load_dotenv(os.path.join(_HERE, ".env"), override=False)
load_dotenv(os.path.join(_HERE, "..", ".env"), override=False)

# ── VideoMAE constants (must match training) ──────────────────────────────────
CLASSES = ["Fighting", "Robbery", "Vandalism", "Normal"]
NUM_CLASSES = len(CLASSES)
CLIP_LEN_MODEL = 16
CLIP_SIZE = 224
VIDEOMAE_MEAN = [0.485, 0.456, 0.406]
VIDEOMAE_STD = [0.229, 0.224, 0.225]
DEFAULT_VIDEOMAE_BASE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"


# ── Model wrapper (must match train_videomae.py VideoMAEFineTuned) ────────────

class VideoMAEFineTuned(nn.Module):
    """VideoMAE with Dropout before classifier — mirrors training architecture."""

    def __init__(self, base: nn.Module, dropout: float = 0.3):
        super().__init__()
        self.videomae = base.videomae
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = base.classifier

    def forward(self, pixel_values: torch.Tensor):
        out = self.videomae(pixel_values)
        pooled = out.last_hidden_state.mean(dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return type("Out", (), {"logits": logits})()


# ── Global model singleton ────────────────────────────────────────────────────
_model: Optional[VideoMAEFineTuned] = None
_device: str = "cpu"
_last_incident_by_camera: Dict[str, float] = {}
_prediction_count_by_camera: Dict[str, int] = {}


def _load_model() -> VideoMAEFineTuned:
    """Load the fine-tuned VideoMAE checkpoint. Called once at startup."""
    global _model, _device

    if _model is not None:
        return _model

    try:
        from transformers import VideoMAEForVideoClassification  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: transformers. Install backend requirements:\n"
            "  pip install -r backend/requirements.txt\n"
            "Or skip running the VideoMAE worker for YOLO-only demo."
        ) from e

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path_raw = os.getenv(
        "VIDEOMAE_WEIGHTS_PATH",
        os.path.join(_HERE, "models", "videomae_best.pth"),
    )
    weights_path = Path(weights_path_raw).expanduser()
    if not weights_path.is_absolute():
        # Resolve relative paths from current cwd first, then fallback to backend dir.
        cwd_candidate = (Path.cwd() / weights_path).resolve()
        backend_candidate = (Path(_HERE) / weights_path).resolve()
        if cwd_candidate.exists():
            weights_path = cwd_candidate
        elif backend_candidate.exists():
            weights_path = backend_candidate
        else:
            weights_path = cwd_candidate

    print(f"[VideoMAE] Loading checkpoint from: {weights_path}")
    print(f"[VideoMAE] Device: {_device}")
    if not weights_path.exists():
        raise FileNotFoundError(
            "VideoMAE checkpoint not found.\n"
            f"Set VIDEOMAE_WEIGHTS_PATH to an existing .pth file.\n"
            f"Current value: {weights_path_raw}\n"
            f"Resolved path: {weights_path}\n"
            f"Checked fallback: {(Path(_HERE) / weights_path_raw).resolve()}"
        )

    # Build the same architecture used during training.
    # Default base is kinetics-finetuned backbone unless overridden.
    base_model_name = os.getenv("VIDEOMAE_BASE_MODEL", DEFAULT_VIDEOMAE_BASE_MODEL)
    inference_dropout = float(os.getenv("VIDEOMAE_DROPOUT", "0.3"))

    base = VideoMAEForVideoClassification.from_pretrained(
        base_model_name,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model = VideoMAEFineTuned(base, dropout=inference_dropout)

    # Load trained weights — the checkpoint has correct 4-class classifier
    ckpt = torch.load(str(weights_path), map_location=_device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Handle possible 'module.' prefix from DataParallel/SWA
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # ── Remap attention bias keys across transformers versions ──────────
    # Older transformers: q_bias, v_bias (no k_bias — key bias is zero)
    # Newer transformers: query.bias, key.bias, value.bias
    # Without this remap, 24 trained attention bias tensors are silently
    # dropped by strict=False, corrupting the attention mechanism.
    remapped = {}
    keys_to_remove = []
    for k, v in state_dict.items():
        if k.endswith(".q_bias"):
            new_key = k.replace(".q_bias", ".query.bias")
            remapped[new_key] = v
            keys_to_remove.append(k)
        elif k.endswith(".v_bias"):
            new_key = k.replace(".v_bias", ".value.bias")
            remapped[new_key] = v
            keys_to_remove.append(k)
    if remapped:
        for k in keys_to_remove:
            del state_dict[k]
        state_dict.update(remapped)
        # Also add zero key.bias for layers that had q_bias/v_bias
        # (the old format had no k_bias — it was implicitly zero)
        for k in list(remapped.keys()):
            if ".query.bias" in k:
                key_bias_name = k.replace(".query.bias", ".key.bias")
                if key_bias_name not in state_dict:
                    state_dict[key_bias_name] = torch.zeros_like(remapped[k])
        print(f"[VideoMAE] 🔄 Remapped {len(remapped)} attention bias keys "
              f"(old→new transformers format)")

    load_res = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(load_res, "missing_keys", []) or [])
    unexpected = list(getattr(load_res, "unexpected_keys", []) or [])

    if missing:
        print(f"[VideoMAE] ⚠️  Missing keys ({len(missing)}): "
              f"{missing[:3]}{' ...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"[VideoMAE] ⚠️  Unexpected keys ({len(unexpected)}): "
              f"{unexpected[:3]}{' ...' if len(unexpected) > 3 else ''}")
    model.to(_device)
    model.eval()

    metrics = ckpt.get("metrics", {})
    epoch = ckpt.get("epoch", "?")
    print(f"[VideoMAE] ✅ Loaded (epoch={epoch}, macro_F1={metrics.get('macro_f1', '?')})")
    print(f"[VideoMAE] Base model: {base_model_name}")
    print(f"[VideoMAE] Inference dropout: {inference_dropout}")
    print(f"[VideoMAE] Classes: {CLASSES}")

    _model = model
    return _model


def _preprocess_frames(frames_bgr: List[np.ndarray]) -> torch.Tensor:
    """
    Convert 16 BGR uint8 frames from OpenCV into the tensor format
    expected by VideoMAE.

    Pipeline (matches training dataset):
      1. Resize to 224×224
      2. BGR → RGB
      3. uint8 → float32 [0, 1]
      4. Normalize with ImageNet mean/std
      5. Stack → (1, 16, 3, 224, 224)
    """
    processed = []
    for frame in frames_bgr[:CLIP_LEN_MODEL]:
        # Resize
        f = cv2.resize(frame, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_LINEAR)
        # BGR → RGB, uint8 → float32 [0, 1]
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        processed.append(f)

    # Pad if fewer than 16 frames (repeat last frame)
    while len(processed) < CLIP_LEN_MODEL:
        processed.append(processed[-1])

    # (16, H, W, 3) → (16, 3, H, W) tensor
    clip = torch.from_numpy(np.stack(processed)).permute(0, 3, 1, 2)

    # Normalize with ImageNet mean/std
    mean = torch.tensor(VIDEOMAE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(VIDEOMAE_STD).view(1, 3, 1, 1)
    clip = (clip - mean) / std

    # Add batch dim → (1, 16, 3, 224, 224)
    return clip.unsqueeze(0)


def run_videomae(frames_16: List[np.ndarray]) -> Tuple[str, float]:
    """
    Run VideoMAE inference on 16 BGR uint8 frames.

    Returns:
      - predicted_class: "Fighting" | "Robbery" | "Vandalism" | "Normal"
      - confidence: float in [0, 1]

    If the top prediction is "Normal", confidence is returned as 0.0
    so the incident handler ignores it (Normal is not an incident).
    """
    model = _load_model()

    clip_tensor = _preprocess_frames(frames_16).to(_device)

    with torch.no_grad(), torch.amp.autocast(_device, enabled=(_device == "cuda")):
        out = model(pixel_values=clip_tensor)
        probs = F.softmax(out.logits, dim=1)  # (1, 4)

    probs_np = probs.cpu().numpy()[0]
    pred_idx = int(np.argmax(probs_np))
    pred_class = CLASSES[pred_idx]
    confidence = float(probs_np[pred_idx])

    # "Normal" is not an incident — return 0.0 confidence to skip
    if pred_class == "Normal":
        return "Normal", 0.0

    print(
        f"[VideoMAE] Prediction: {pred_class} ({confidence:.1%}) | "
        f"all: {', '.join(f'{c}={p:.1%}' for c, p in zip(CLASSES, probs_np))}"
    )

    return pred_class, confidence


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def discover_stream_keys(r: redis.Redis) -> List[str]:
    keys: List[str] = []
    cursor = 0
    while True:
        cursor, batch = r.scan(cursor=cursor, match="camera:*:clips", count=200)
        keys.extend([k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k) for k in batch])
        if cursor == 0:
            break
    return sorted(set(keys))


def main() -> None:
    # Pre-load the VideoMAE model at startup
    _load_model()

    supabase_url = _env("SUPABASE_URL")
    supabase_key = _env("SUPABASE_SERVICE_ROLE_KEY")
    supabase = create_client(supabase_url, supabase_key)

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    r = redis.Redis.from_url(redis_url)
    redis_retry_s = float(os.getenv("REDIS_RETRY_SECONDS", "2"))

    def ensure_redis() -> None:
        while True:
            try:
                r.ping()
                print(f"[worker] Connected to Redis at {redis_url}")
                return
            except RedisConnectionError:
                print(f"[worker] Redis unavailable at {redis_url}; retrying in {redis_retry_s:.1f}s...")
                time.sleep(redis_retry_s)

    ensure_redis()

    min_conf = float(os.getenv("VIDEOMAE_MIN_CONF", "0.60"))
    incident_cooldown_s = float(os.getenv("INCIDENT_COOLDOWN_SECONDS", "60"))
    refresh_s = float(os.getenv("STREAM_DISCOVERY_REFRESH_S", "10"))
    # Number of initial predictions per camera to discard (warm-up).
    # The first clips are built from a barely-filled frame buffer and
    # produce unreliable outlier predictions.
    warmup_skip = int(os.getenv("VIDEOMAE_WARMUP_SKIP", "2"))

    last_ids: Dict[str, str] = {}
    next_discovery = 0.0
    streams: List[str] = []

    while True:
        try:
            now = time.time()
            if now >= next_discovery:
                streams = discover_stream_keys(r)
                for s in streams:
                    last_ids.setdefault(s, "$")
                next_discovery = now + refresh_s

            if not streams:
                time.sleep(0.5)
                continue

            # Build XREAD dict
            xread_args = {s: last_ids.get(s, "$") for s in streams}
            resp = r.xread(xread_args, block=5_000, count=10)
            if not resp:
                continue

            for stream_key, messages in resp:
                sk = stream_key.decode("utf-8") if isinstance(stream_key, (bytes, bytearray)) else str(stream_key)
                for msg_id, data in messages:
                    mid = msg_id.decode("utf-8") if isinstance(msg_id, (bytes, bytearray)) else str(msg_id)
                    last_ids[sk] = mid

                    try:
                        camera_id = _b(data.get(b"camera_id") or data.get("camera_id"))
                        camera_name = _b(data.get(b"camera_name") or data.get("camera_name"))
                        camera_location = _b(data.get(b"camera_location") or data.get("camera_location"))
                        frames_16 = pickle.loads(data.get(b"frames") or data.get("frames"))
                        frames_30s = pickle.loads(data.get(b"frames_30s") or data.get("frames_30s") or pickle.dumps([]))
                        fps = float(_b(data.get(b"fps") or data.get("fps") or "8"))
                    except Exception:
                        continue

                    predicted_class, confidence = run_videomae(frames_16)

                    # Skip early predictions — buffer hasn't stabilised yet.
                    n_seen = _prediction_count_by_camera.get(camera_id, 0)
                    _prediction_count_by_camera[camera_id] = n_seen + 1
                    if n_seen < warmup_skip:
                        print(
                            f"[worker] warm-up skip {n_seen + 1}/{warmup_skip} "
                            f"camera={camera_id} class={predicted_class} "
                            f"conf={confidence:.3f}"
                        )
                        continue

                    if confidence < min_conf:
                        print(
                            f"[worker] below threshold camera={camera_id} class={predicted_class} "
                            f"conf={confidence:.3f} < {min_conf:.3f}"
                        )
                        continue

                    # Deduplicate repeated alerts from the same camera within cooldown window.
                    now_ts = time.time()
                    last_ts = _last_incident_by_camera.get(camera_id, 0.0)
                    if now_ts - last_ts < incident_cooldown_s:
                        print(
                            f"[worker] incident suppressed camera={camera_id} "
                            f"cooldown={incident_cooldown_s:.0f}s remaining={incident_cooldown_s - (now_ts - last_ts):.1f}s"
                        )
                        continue

                    try:
                        # If the producer skipped 30s frames (demo/perf mode), fall back to frames_16.
                        if not frames_30s:
                            frames_30s = frames_16
                        clip_url, twilio_status = handle_incident(
                            supabase=supabase,
                            camera_id=camera_id,
                            camera_name=camera_name,
                            camera_location=camera_location,
                            predicted_class=predicted_class,
                            confidence=confidence,
                            frames_30s=frames_30s,
                            fps=fps,
                        )
                        _last_incident_by_camera[camera_id] = time.time()
                        print(
                            f"[worker] incident handled camera={camera_id} class={predicted_class} "
                            f"conf={confidence:.3f} clip_url={clip_url} twilio_status={twilio_status}"
                        )
                    except Exception as e:
                        print(f"[worker] incident handling failed camera={camera_id}: {e}")
                        continue
        except RedisConnectionError:
            print(f"[worker] Lost Redis connection; retrying in {redis_retry_s:.1f}s...")
            time.sleep(redis_retry_s)
            ensure_redis()


def _b(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore")
    return str(v)


if __name__ == "__main__":
    main()

