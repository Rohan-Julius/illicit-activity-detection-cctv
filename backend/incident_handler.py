from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from supabase import Client
from twilio.rest import Client as TwilioClient


def _format_twilio_address(channel: str, raw_number: str) -> str:
    v = (raw_number or "").strip()
    if not v:
        return v
    if channel == "whatsapp":
        return v if v.startswith("whatsapp:") else f"whatsapp:{v}"
    # sms/mms fallback
    return v.replace("whatsapp:", "")


def _open_video_writer(out_path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    """
    Try codecs in reliability order for browser/WhatsApp playback.
    avc1/h264 is preferred; fallback to mp4v.
    """
    for fourcc_name in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        vw = cv2.VideoWriter(out_path, fourcc, float(max(fps, 1.0)), (w, h))
        if vw.isOpened():
            return vw
        vw.release()
    raise RuntimeError("Unable to open MP4 writer with avc1/H264/mp4v codecs")


def save_mp4(frames_bgr: List[np.ndarray], fps: float, out_path: str) -> None:
    if not frames_bgr:
        raise ValueError("No frames to save")
    # Ensure even dimensions for codec compatibility and avoid green/distorted decode issues.
    h, w = frames_bgr[0].shape[:2]
    h = h - (h % 2)
    w = w - (w % 2)
    vw = _open_video_writer(out_path, fps=fps, w=w, h=h)
    try:
        for f in frames_bgr:
            if f is None:
                continue
            # Normalize shape/channels/types before encoding.
            if f.ndim == 2:
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            elif f.ndim == 3 and f.shape[2] == 4:
                f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)

            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            # Guard against non-contiguous memory artifacts in encoder.
            vw.write(np.ascontiguousarray(f))
    finally:
        vw.release()


def upload_clip_and_get_url(supabase: Client, bucket: str, camera_id: str, ts: str, clip_path: str) -> str:
    key = f"{camera_id}/{ts}.mp4"
    with open(clip_path, "rb") as f:
        # supabase-py expects storage file options as strings in headers-like payload.
        supabase.storage.from_(bucket).upload(
            key,
            f,
            {"content-type": "video/mp4", "upsert": "true"},
        )
    return supabase.storage.from_(bucket).get_public_url(key)


def insert_incident(
    supabase: Client,
    *,
    camera_id: str,
    camera_name: str,
    predicted_class: str,
    confidence: float,
    clip_url: Optional[str],
    timestamp: Optional[str] = None,
) -> Any:
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    canonical = str(predicted_class).strip()
    title_map = {
        "fighting": "Fighting",
        "robbery": "Robbery",
        "vandalism": "Vandalism",
    }
    title_value = title_map.get(canonical.lower(), canonical)
    lower_value = title_value.lower()

    base_payload = {
        "camera_id": camera_id,
        "camera_name": camera_name,
        "confidence": confidence,
        "timestamp": ts,
        "clip_url": clip_url,
        "twilio_status": "pending",
        "acknowledged": False,
    }

    # Try title case first (new schema), then lowercase fallback (older constraint variants).
    try:
        payload = {**base_payload, "class": title_value}
        return supabase.table("incidents").insert(payload).execute()
    except Exception as e:
        text = str(e)
        if "class_check" in text or "violates check constraint" in text:
            payload = {**base_payload, "class": lower_value}
            return supabase.table("incidents").insert(payload).execute()
        raise


def update_twilio_status_by_clip_url(supabase: Client, clip_url: str, status: str) -> None:
    supabase.table("incidents").update({"twilio_status": status}).eq("clip_url", clip_url).execute()


def update_twilio_status_by_camera_timestamp(
    supabase: Client, camera_id: str, ts_iso: str, status: str
) -> None:
    supabase.table("incidents").update({"twilio_status": status}).eq("camera_id", camera_id).eq("timestamp", ts_iso).execute()


def set_camera_alert(supabase: Client, camera_id: str) -> None:
    supabase.table("cameras").update({"status": "alert"}).eq("id", camera_id).execute()


def send_twilio_message(
    *,
    account_sid: str,
    auth_token: str,
    to_addr: str,
    from_addr: str,
    body: str,
    media_url: str,
) -> Optional[str]:
    twilio = TwilioClient(account_sid, auth_token)
    twilio_channel = os.getenv("TWILIO_CHANNEL", "whatsapp").strip().lower()
    kwargs = {
        "to": to_addr,
        "from_": from_addr,
        "body": body,
    }
    include_media = os.getenv("TWILIO_INCLUDE_MEDIA", "1").strip().lower() not in ("0", "false", "no")
    # WhatsApp commonly rejects media payloads due to channel/content constraints (e.g., 63021).
    # Default to text + URL for WhatsApp; only attach media for non-WhatsApp channels.
    if twilio_channel == "whatsapp" and media_url:
        kwargs["body"] = f"{body}\nClip: {media_url}"
    elif media_url and include_media:
        kwargs["media_url"] = [media_url]
    msg = twilio.messages.create(**kwargs)
    return getattr(msg, "sid", None)


def handle_incident(
    *,
    supabase: Client,
    camera_id: str,
    camera_name: str,
    camera_location: str,
    predicted_class: str,
    confidence: float,
    frames_30s: List[np.ndarray],
    fps: float,
) -> Tuple[Optional[str], str]:
    """
    Returns (clip_url, twilio_status).
    """
    bucket = os.getenv("SUPABASE_CLIPS_BUCKET", "clips")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ts_iso = datetime.now(timezone.utc).isoformat()
    clip_url: Optional[str] = None

    try:
        with tempfile.TemporaryDirectory() as td:
            clip_path = os.path.join(td, f"{camera_id}_{ts}.mp4")
            save_mp4(frames_30s, fps=fps, out_path=clip_path)
            clip_url = upload_clip_and_get_url(supabase, bucket=bucket, camera_id=camera_id, ts=ts, clip_path=clip_path)
    except Exception as e:
        print(f"[incident] clip upload failed for camera={camera_id}: {e}")

    try:
        insert_incident(
            supabase,
            camera_id=camera_id,
            camera_name=camera_name,
            predicted_class=predicted_class,
            confidence=confidence,
            clip_url=clip_url,
            timestamp=ts_iso,
        )
    except Exception as e:
        print(f"[incident] DB insert failed camera={camera_id}: {e}")
        raise

    twilio_status = "pending"
    try:
        sid = None
        # Twilio config guard: skip if creds/addresses missing.
        # Default to WhatsApp channel (user requirement), override with TWILIO_CHANNEL=sms if needed.
        twilio_channel = os.getenv("TWILIO_CHANNEL", "whatsapp").strip().lower()
        sid_val = os.getenv("TWILIO_ACCOUNT_SID", "")
        token_val = os.getenv("TWILIO_AUTH_TOKEN", "")
        to_raw = os.getenv("ALERT_TO_PHONE_NUMBER", "")
        from_raw = os.getenv("TWILIO_WHATSAPP_FROM", "") or os.getenv("TWILIO_PHONE_NUMBER", "")

        to_val = _format_twilio_address(twilio_channel, to_raw)
        from_val = _format_twilio_address(twilio_channel, from_raw)

        if sid_val and token_val and to_val and from_val:
            sid = send_twilio_message(
                account_sid=sid_val,
                auth_token=token_val,
                to_addr=to_val,
                from_addr=from_val,
                body=(
                    f"ALERT: {predicted_class} detected\n"
                    f"Camera: {camera_name}\n"
                    f"Location: {camera_location}\n"
                    f"Confidence: {confidence:.0%}\n"
                    f"Time: {datetime.now(timezone.utc).isoformat()}"
                ),
                media_url=clip_url or "",
            )
            print(f"[incident] Twilio accepted message sid={sid} channel={twilio_channel} to={to_val}")
        else:
            print("[incident] Twilio credentials/addresses missing; skipping send")

        twilio_status = "sent" if sid else "failed"
    except Exception as e:
        print(f"[incident] Twilio send failed camera={camera_id}: {e}")
        twilio_status = "failed"

    try:
        if clip_url:
            update_twilio_status_by_clip_url(supabase, clip_url=clip_url, status=twilio_status)
        else:
            update_twilio_status_by_camera_timestamp(
                supabase, camera_id=camera_id, ts_iso=ts_iso, status=twilio_status
            )
    except Exception as e:
        print(f"[incident] Twilio status update failed camera={camera_id}: {e}")

    try:
        set_camera_alert(supabase, camera_id=camera_id)
    except Exception as e:
        print(f"[incident] Camera status update failed camera={camera_id}: {e}")

    return clip_url, twilio_status

