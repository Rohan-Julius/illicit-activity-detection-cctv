"""
setup_demo.py — Update the first camera in Supabase to use test.mp4 for the demo.

Usage:
    python setup_demo.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env", override=False)
load_dotenv(_HERE.parent / ".env", override=False)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Resolve the absolute path to test.mp4
test_video = str(_HERE / "test.mp4")
if not Path(test_video).exists():
    raise FileNotFoundError(f"test.mp4 not found at: {test_video}")

print(f"[setup_demo] test.mp4 path: {test_video}")

# Get all cameras
cameras = supabase.table("cameras").select("*").order("created_at").execute()

if not cameras.data:
    print("[setup_demo] No cameras found in the database!")
    print("[setup_demo] Run the SQL in scripts/setup-supabase.sql first.")
    exit(1)

print(f"[setup_demo] Found {len(cameras.data)} cameras:")
for cam in cameras.data:
    print(f"  - {cam['name']} ({cam['status']}) → {cam.get('video_url', 'None')[:60]}...")

# Update the first camera to use test.mp4 and stream_type=websocket
first_cam = cameras.data[0]
supabase.table("cameras").update({
    "video_url": test_video,
    "status": "live",
    "stream_type": "websocket",
}).eq("id", first_cam["id"]).execute()

print(f"\n[setup_demo] ✅ Updated '{first_cam['name']}' (id={first_cam['id']}):")
print(f"  video_url  → {test_video}")
print(f"  status     → live")
print(f"  stream_type → websocket")
print()
print("[setup_demo] Ready for demo! Start the services:")
print("  1. redis-server")
print("  2. cd backend && uvicorn app:app --host 0.0.0.0 --port 8000")
print("  3. cd backend && python worker_videomae.py")
print("  4. pnpm dev  (from project root)")
