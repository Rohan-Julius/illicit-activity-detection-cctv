from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import cv2
import numpy as np
import pickle
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import redis
from supabase import create_client
from ultralytics import YOLO
from collections import deque

from models.zerodce import ZeroDCEConfig, load_zerodce
from utils import encode_jpeg, mean_brightness_bgr, parse_opencv_source

# Load env from backend/.env (preferred) and repo root .env (fallback)
_HERE = os.path.dirname(__file__)
load_dotenv(os.path.join(_HERE, ".env"), override=False)
load_dotenv(os.path.join(_HERE, "..", ".env"), override=False)


@dataclass
class CameraRow:
    id: str
    name: str
    location: str
    status: str
    video_url: Optional[str]
    stream_type: str = "websocket"


class CameraHub:
    def __init__(self) -> None:
        self._clients: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def add(self, camera_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.setdefault(camera_id, []).append(ws)

    async def remove(self, camera_id: str, ws: WebSocket) -> None:
        async with self._lock:
            arr = self._clients.get(camera_id, [])
            self._clients[camera_id] = [x for x in arr if x is not ws]
            if not self._clients[camera_id]:
                self._clients.pop(camera_id, None)

    async def broadcast(self, camera_id: str, jpeg_bytes: bytes) -> None:
        async with self._lock:
            targets = list(self._clients.get(camera_id, []))
        if not targets:
            return
        dead: List[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_bytes(jpeg_bytes)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.remove(camera_id, ws)


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing env var: {name}")
    return v


app = FastAPI()
hub = CameraHub()

_supabase = None
_yolo = None
_zerodce = None
_redis: Optional[redis.Redis] = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
async def startup() -> None:
    global _supabase, _yolo, _zerodce

    supabase_url = _env("SUPABASE_URL")
    supabase_key = _env("SUPABASE_SERVICE_ROLE_KEY")
    _supabase = create_client(supabase_url, supabase_key)

    global _redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    _redis = redis.Redis.from_url(redis_url)

    yolo_weights_path = os.getenv("YOLO_WEIGHTS_PATH", "models/yolo11s.pt")
    # Resolve relative paths from backend directory
    if not os.path.isabs(yolo_weights_path):
        yolo_weights_path = os.path.join(os.path.dirname(__file__), yolo_weights_path)
    _yolo = YOLO(yolo_weights_path)

    zerodce_weights = os.getenv("ZERODCE_WEIGHTS_PATH", "models/stage2_best.pth")
    # Resolve relative paths from backend directory
    if zerodce_weights and not os.path.isabs(zerodce_weights):
        zerodce_weights = os.path.join(os.path.dirname(__file__), zerodce_weights)
    if zerodce_weights and os.path.exists(zerodce_weights):
        _zerodce = load_zerodce(ZeroDCEConfig(weights_path=zerodce_weights, device=_device))
    else:
        _zerodce = None

    # Auto-startup disabled - cameras only stream when client connects via WebSocket
    # To enable auto-startup, set environment variable AUTO_START_CAMERAS=1
    auto_start = os.getenv("AUTO_START_CAMERAS", "0").lower() not in ("0", "false", "no")
    
    if auto_start:
        cameras = _supabase.table("cameras").select("*").eq("status", "alert").execute()
        print(f"[startup] launching workers for {len(cameras.data or [])} active cameras")
        for row in cameras.data or []:
            cam = CameraRow(
                id=str(row["id"]),
                name=str(row.get("name") or ""),
                location=str(row.get("location") or ""),
                status=str(row.get("status") or "live"),
                video_url=row.get("video_url"),
                stream_type=str(row.get("stream_type") or "websocket"),
            )
            asyncio.create_task(camera_worker(cam))
    else:
        print("[startup] AUTO_START_CAMERAS disabled - cameras stream only on WebSocket connect")


def _maybe_enhance_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Synchronous: dark-frame enhancement. Run in thread pool."""
    if os.getenv("ZERODCE_ENABLED", "1").lower() in ("0", "false", "no"):
        return frame_bgr
    if _zerodce is None:
        return frame_bgr
    b = mean_brightness_bgr(frame_bgr)
    thresh = float(os.getenv("DARK_BRIGHTNESS_THRESHOLD", "0.35"))
    if b < thresh:
        return enhance_with_zerodce(frame_bgr)
    return frame_bgr


def _prepare_and_enqueue_clip_job(
    *,
    cam: CameraRow,
    frames_16: List[np.ndarray],
    frames_30s: List[np.ndarray],
    target_fps: float,
    stride: int,
    include_30s_in_redis: bool,
) -> None:
    """
    Heavy clip-serialization path moved off the asyncio loop.
    Downscales frames before pickling to avoid stalls during dense action scenes.
    """
    assert _redis is not None
    clip_size = max(int(os.getenv("CLIP_FRAME_SIZE", "224")), 64)

    # Match training resize behavior as closely as possible.
    frames_16_small = [cv2.resize(f, (clip_size, clip_size), interpolation=cv2.INTER_LINEAR) for f in frames_16]
    payload = {
        "camera_id": cam.id,
        "camera_name": cam.name,
        "camera_location": cam.location,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frames": pickle.dumps(frames_16_small, protocol=pickle.HIGHEST_PROTOCOL),
        "fps": str(target_fps / stride),
    }

    if include_30s_in_redis:
        frames_30s_small = [
            cv2.resize(f, (clip_size, clip_size), interpolation=cv2.INTER_LINEAR) for f in frames_30s
        ]
        payload["frames_30s"] = pickle.dumps(frames_30s_small, protocol=pickle.HIGHEST_PROTOCOL)

    _redis.xadd(f"camera:{cam.id}:clips", payload, maxlen=500, approximate=True)


async def camera_worker(cam: CameraRow) -> None:
    assert _supabase is not None
    assert _yolo is not None
    assert _redis is not None

    src = parse_opencv_source(cam.video_url)
    cap: Optional[cv2.VideoCapture] = None

    # Keep a simple fps cap to reduce CPU in dev.
    target_fps = float(os.getenv("TARGET_FPS", "12"))
    frame_interval = 1.0 / max(target_fps, 1.0)
    # First N frames: JPEG only (no YOLO) so the browser gets pixels quickly while models warm up.
    fast_preview = max(int(os.getenv("FAST_PREVIEW_FRAMES", "8")), 0)
    frame_index = 0
    # Run expensive YOLO only every Nth frame; draw cached boxes on other frames (smooth video on CPU).
    yolo_every_n = max(int(os.getenv("YOLO_EVERY_N_FRAMES", "2")), 1)
    last_xyxy: Optional[np.ndarray] = None
    last_confs: Optional[np.ndarray] = None

    # Buffers
    # Keep a source-frame history so VideoMAE clips can mirror training stride-4 construction:
    # 16 model frames sampled from 64 source frames.
    clip_len = 16
    videomae_stride = max(int(os.getenv("VIDEOMAE_TEMPORAL_STRIDE", "4")), 1)
    src_frames_needed = clip_len * videomae_stride
    buf_src: Deque[np.ndarray] = deque(maxlen=src_frames_needed)
    buf30s: Deque[np.ndarray] = deque(maxlen=int(max(target_fps, 1.0) * 30))
    consecutive_person = 0
    last_enqueue_t = 0.0
    enqueue_cooldown_s = float(os.getenv("CLIP_ENQUEUE_COOLDOWN_S", "3.0"))
    clip_enqueue_enabled = os.getenv("CLIP_ENQUEUE_ENABLED", "0").lower() not in ("0", "false", "no")
    include_30s_in_redis = os.getenv("CLIP_INCLUDE_30S_IN_REDIS", "0").lower() not in ("0", "false", "no")

    while True:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                print(f"[camera_worker] Failed to open source for {cam.id}: {cam.video_url!r}; retrying...")
                await asyncio.sleep(1.5)
                continue

        t0 = asyncio.get_event_loop().time()
        ok, frame = cap.read()
        if not ok or frame is None:
            # Loop video for demo (pre-recorded MP4 files)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_index = 0
            await asyncio.sleep(0.05)
            continue

        frame_index += 1

        # Update last_seen_at periodically
        if int(t0) % 10 == 0:
            try:
                _supabase.table("cameras").update(
                    {"last_seen_at": datetime.now(timezone.utc).isoformat()}
                ).eq("id", cam.id).execute()
            except Exception:
                pass

        # Heavy work must NOT run on the asyncio event loop — it blocks WebSocket handshakes and delivery.
        if frame_index <= fast_preview:
            working = await asyncio.to_thread(_maybe_enhance_frame, frame)
            buf30s.append(working)
            try:
                jpeg = await asyncio.to_thread(
                    encode_jpeg, working, int(os.getenv("JPEG_QUALITY", "75"))
                )
                await hub.broadcast(cam.id, jpeg)
            except Exception:
                pass
            dt = asyncio.get_event_loop().time() - t0
            if dt < frame_interval:
                await asyncio.sleep(frame_interval - dt)
            continue

        # Enhance if dark (in thread)
        working = await asyncio.to_thread(_maybe_enhance_frame, frame)
        buf_src.append(working)

        # Track 30s buffer (store BGR frames)
        buf30s.append(working)

        # YOLO: full infer every yolo_every_n frames; otherwise reuse last boxes on current frame (video keeps moving).
        sub_i = frame_index - fast_preview  # 1,2,3,... after preview
        run_yolo_now = sub_i % yolo_every_n == 1 or last_xyxy is None

        if run_yolo_now:
            annotated, has_person, xyxy, confs = await asyncio.to_thread(
                run_yolo_infer_person_boxes, working
            )
            last_xyxy, last_confs = xyxy, confs
        else:
            xyxy = last_xyxy
            confs = last_confs
            if xyxy is not None and len(xyxy) > 0:
                annotated = await asyncio.to_thread(draw_person_boxes_on_frame, working, xyxy, confs)
            else:
                annotated = working.copy()
            has_person = xyxy is not None and len(xyxy) > 0

        if has_person:
            consecutive_person += 1
        else:
            consecutive_person = 0

        # Enqueue 16-frame clip job when buffer full (optional; can be expensive on CPU)
        if (
            clip_enqueue_enabled
            and len(buf_src) >= src_frames_needed
            and consecutive_person >= 3  # Wait for 3 consecutive person frames before enqueuing
            and (t0 - last_enqueue_t) >= enqueue_cooldown_s
        ):
            last_enqueue_t = t0
            try:
                src_list = list(buf_src)
                # Build a 16-frame clip from the latest temporal window, matching training stride.
                frames_16 = src_list[-src_frames_needed::videomae_stride]
                if len(frames_16) < clip_len:
                    frames_16.extend([frames_16[-1]] * (clip_len - len(frames_16)))
                elif len(frames_16) > clip_len:
                    frames_16 = frames_16[:clip_len]
                stride = max(int(os.getenv("CLIP_30S_STRIDE", "6")), 1)
                frames_30s = list(buf30s)[::stride] if include_30s_in_redis else []
                await asyncio.to_thread(
                    _prepare_and_enqueue_clip_job,
                    cam=cam,
                    frames_16=frames_16,
                    frames_30s=frames_30s,
                    target_fps=target_fps,
                    stride=stride,
                    include_30s_in_redis=include_30s_in_redis,
                )
            except Exception:
                pass

        # Broadcast
        try:
            jpeg = await asyncio.to_thread(
                encode_jpeg, annotated, int(os.getenv("JPEG_QUALITY", "75"))
            )
            await hub.broadcast(cam.id, jpeg)
        except Exception:
            pass

        # FPS cap
        dt = asyncio.get_event_loop().time() - t0
        if dt < frame_interval:
            await asyncio.sleep(frame_interval - dt)


def enhance_with_zerodce(frame_bgr: np.ndarray) -> np.ndarray:
    assert _zerodce is not None

    # BGR uint8 -> RGB float tensor in [0,1]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = x.to(_device)
    with torch.no_grad():
        y = _zerodce(x)
    y = y.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    out_rgb = (y * 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr


def run_yolo_annotate(frame_bgr: np.ndarray) -> np.ndarray:
    assert _yolo is not None

    # Ultralytics expects RGB images typically; it will convert internally for numpy arrays,
    # but we keep it explicit for consistent colors.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _yolo.predict(rgb, verbose=False, conf=float(os.getenv("YOLO_CONF", "0.25")))
    if not results:
        return frame_bgr
    plotted = results[0].plot()
    # plot() returns BGR uint8
    return plotted


def draw_person_boxes_on_frame(
    frame_bgr: np.ndarray, xyxy: np.ndarray, confs: np.ndarray
) -> np.ndarray:
    """Draw cached person boxes (same coords as frame size)."""
    annotated = frame_bgr.copy()
    if xyxy is None or len(xyxy) == 0:
        return annotated
    for (x1, y1, x2, y2), conf in zip(xyxy, confs):
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        label = f"person {float(conf):.2f}"
        cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1i, max(0, y1i - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_yolo_infer_person_boxes(frame_bgr: np.ndarray) -> Tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
    """
    Person-only YOLO. Returns (annotated_bgr, has_person, xyxy Nx4, confs N).

    COCO: class 0 == person. Custom datasets may differ — set YOLO_PERSON_CLASS_IDS=0
    """
    assert _yolo is not None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    imgsz = int(os.getenv("YOLO_IMGSZ", "640"))
    results = _yolo.predict(
        rgb,
        verbose=False,
        conf=float(os.getenv("YOLO_CONF", "0.35")),
        iou=float(os.getenv("YOLO_IOU", "0.45")),
        classes=[0],  # person only (COCO)
        imgsz=imgsz,
    )
    if not results:
        return frame_bgr.copy(), False, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None or boxes.xyxy is None or boxes.conf is None:
        return frame_bgr.copy(), False, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    has_person = xyxy.shape[0] > 0
    annotated = draw_person_boxes_on_frame(frame_bgr, xyxy, confs)
    return annotated, has_person, xyxy, confs


@app.websocket("/ws/camera/{camera_id}")
async def ws_camera(websocket: WebSocket, camera_id: str) -> None:
    await websocket.accept()
    await hub.add(camera_id, websocket)
    try:
        while True:
            # Keep connection alive; server pushes frames.
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await hub.remove(camera_id, websocket)

