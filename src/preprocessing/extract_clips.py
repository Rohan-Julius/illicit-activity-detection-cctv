"""
extract_clips.py
===============================================================
4 classes: Fighting, Robbery, Vandalism, Normal  (Assault removed)

WHAT THIS SCRIPT DOES:
  Extracts 16-frame clips from UCF-Crime videos using the official
  Action_Recognition_splits (train_001.txt – test_004.txt).

  For each video in each fold's train/test split:
    1. Decode frames at 12 FPS with Zero-DCE++ enhancement
    2. Divide into CLIPS_PER_VIDEO equal segments
    3. Sample one stride-4 clip per segment (16 frames × stride 4 = 64 src frames ≈ 5.3s)
    4. Deduplicate using motion fingerprints (threshold 0.99)
    5. Save as (16, 224, 224, 3) float32 .npy files

OUTPUT STRUCTURE:
  data/processed/clips/
    fold_1/train/Fighting/scene/*.npy
    fold_1/train/Robbery/scene/*.npy
    fold_1/train/Vandalism/scene/*.npy
    fold_1/train/Normal/scene/*.npy
    fold_1/test/...
    fold_2/ ...  fold_3/ ...  fold_4/ ...

"""

import sys
import re
import gc
import random
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.infer_zerodce import ZeroDCEInference

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
UCF_DIR    = ROOT / 'data' / 'raw' / 'ucf_crime'
VIDEOS_DIR = UCF_DIR / 'videos'
SPLITS_DIR = UCF_DIR / 'Action_Recognition_splits'
CLIPS_DIR  = ROOT / 'data' / 'processed' / 'clips'
ZERODCE_W  = ROOT / 'models' / 'zerodce' / 'stage2_best.pth'

# ── 4-class setup — Assault removed ───────────────────────────────────────────
UCF_FOLDER_TO_LABEL = {
    'Fighting'           : 'Fighting',
    'Robbery'            : 'Robbery',
    'Vandalism'          : 'Vandalism',
    'Normal_Videos_event': 'Normal',
}

LABELS       = ['Fighting', 'Robbery', 'Vandalism', 'Normal']
VALID_FOLDS  = [1, 2, 3, 4]
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.MP4', '.AVI', '.MKV'}

# ── Clip parameters ────────────────────────────────────────────────────────────
TARGET_FPS        = 12
CLIP_LEN          = 16      # frames stored — VideoMAE-Base native input
STRIDE            = 4       # every 4th src frame → 64 src frames ≈ 5.3s
CLIP_SIZE         = (224, 224)
CLIPS_PER_VIDEO   = {'train': 16, 'test': 8}
CLASS_CAPS        = {lbl: {'train': 9999, 'test': 999} for lbl in LABELS}
DEDUP_THRESHOLD   = 0.99
TEMPORAL_JITTER   = 3
MIN_FRAMES_NEEDED = CLIP_LEN * STRIDE   # = 64 source frames


# ═══════════════════════════════════════════════════════════════════════════════
# Official Split Loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_official_fold(fold: int) -> dict:
    """
    Parse Action_Recognition_splits/train_NNN.txt and test_NNN.txt.
    Returns {'train': {'Fighting': [(folder, stem), ...], ...}, 'test': {...}}
    Folder assignment done by scanning VIDEOS_DIR on disk — not the .txt label
    column which varies between dataset versions.
    """
    assert fold in VALID_FOLDS, f"fold must be one of {VALID_FOLDS}"

    train_file = SPLITS_DIR / f'train_{fold:03d}.txt'
    test_file  = SPLITS_DIR / f'test_{fold:03d}.txt'

    if not SPLITS_DIR.exists():
        raise FileNotFoundError(
            f"Action_Recognition_splits not found: {SPLITS_DIR}\n"
            f"Place at {UCF_DIR}/Action_Recognition_splits/"
        )
    for f in [train_file, test_file]:
        if not f.exists():
            raise FileNotFoundError(f"Split file missing: {f}")

    print(f"\n[Fold {fold}] Building stem→folder lookup from {VIDEOS_DIR}")
    stem_to_folder: dict = {}
    for ucf_folder in UCF_FOLDER_TO_LABEL:
        folder_dir = VIDEOS_DIR / ucf_folder
        if not folder_dir.exists():
            print(f"  [WARN] Not found on disk: {folder_dir}")
            continue
        for p in folder_dir.iterdir():
            if p.suffix in VIDEO_EXTENSIONS:
                stem_to_folder[p.stem] = ucf_folder

    result = {'train': defaultdict(list), 'test': defaultdict(list)}
    skipped_other = skipped_missing = 0

    for split_key, txt_file in [('train', train_file), ('test', test_file)]:
        with open(txt_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fname      = line.split()[0]
                stem       = Path(fname).stem
                ucf_folder = stem_to_folder.get(stem)
                if ucf_folder is None:
                    skipped_missing += 1
                    continue
                label = UCF_FOLDER_TO_LABEL.get(ucf_folder)
                if label is None:
                    skipped_other += 1   # Assault etc. → silently ignored
                    continue
                result[split_key][label].append((ucf_folder, stem))

    result = {'train': dict(result['train']), 'test': dict(result['test'])}

    # Leakage check
    train_stems = {s for v in result['train'].values() for _, s in v}
    test_stems  = {s for v in result['test'].values()  for _, s in v}
    overlap = train_stems & test_stems
    if overlap:
        raise RuntimeError(
            f"Fold {fold}: {len(overlap)} stems appear in BOTH splits.\n"
            f"First 5: {sorted(overlap)[:5]}"
        )

    print(f"[Fold {fold}] Split loaded ✅  "
          f"(skipped {skipped_other} non-target, {skipped_missing} missing)")
    print(f"\n  {'Label':10s}  {'Train':>5}  {'Test':>5}")
    print(f"  {'-'*28}")
    for label in LABELS:
        tr = len(result['train'].get(label, []))
        te = len(result['test'].get(label, []))
        print(f"  {label:10s}  {tr:5d}  {te:5d}")
    print(f"  ✅ No train/test leakage\n")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Clip Deduplication
# ═══════════════════════════════════════════════════════════════════════════════

def deduplicate_clips(clips: list, threshold: float = DEDUP_THRESHOLD) -> list:
    """
    Motion-fingerprint dedup. Uses mean absolute frame-diff — not mean frame,
    which is nearly identical for all clips from fixed-camera CCTV footage.
    """
    if len(clips) <= 1:
        return clips

    def fp(clip: np.ndarray) -> np.ndarray:
        diffs = np.abs(np.diff(clip, axis=0))
        f     = diffs.mean(axis=0).flatten()
        return f / (np.linalg.norm(f) + 1e-8)

    nfps = [fp(c) for c in clips]
    kept = [0]
    for i in range(1, len(clips)):
        if not any(float(np.dot(nfps[i], nfps[j])) > threshold for j in kept):
            kept.append(i)
    return [clips[k] for k in kept]


# ═══════════════════════════════════════════════════════════════════════════════
# Scene Clip Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class SceneClipExtractor:
    """
    Extracts 16-frame clips via uniform segment division with stride-4 sampling.
    Zero-DCE++ low-light enhancement applied per frame during loading.
    OOM-safe: MemoryError caught in _valid() and _build_clip_strided().
    """

    def __init__(self, device: str = 'cuda'):
        print("[Extractor] Loading Zero-DCE++ ...")
        self.enhancer = ZeroDCEInference(
            weights_path         = str(ZERODCE_W),
            device               = device,
            brightness_threshold = 50,
            output_size          = 640,
            enhance_size         = 256,
        )
        print(f"[Extractor] Ready (v12: 4 classes | CLIP_LEN={CLIP_LEN} | "
              f"STRIDE={STRIDE})\n")

    def extract(self, video_path: Path, label: str, split: str,
                fold: int, clip_counter: dict = None) -> dict:
        per_vid   = CLIPS_PER_VIDEO[split]
        class_cap = CLASS_CAPS[label][split]

        if clip_counter and clip_counter.get(label, 0) >= class_cap:
            return {'clips': 0, 'skipped': True}

        out_dir = CLIPS_DIR / f'fold_{fold}' / split / label / 'scene'
        out_dir.mkdir(parents=True, exist_ok=True)

        frames = self._load_frames(video_path)
        n_src  = len(frames)

        if n_src < MIN_FRAMES_NEEDED:
            return {
                'clips' : 0,
                'reason': f'too_short ({n_src} frames, need {MIN_FRAMES_NEEDED})',
            }

        candidates   = self._make_clips(frames, per_vid)
        before_dedup = len(candidates)

        del frames
        gc.collect()

        candidates  = deduplicate_clips(candidates)
        after_dedup = len(candidates)

        if clip_counter is not None:
            remaining  = max(0, class_cap - clip_counter.get(label, 0))
            candidates = candidates[:min(per_vid, remaining)]
        else:
            candidates = candidates[:per_vid]

        n = self._save(candidates, out_dir, video_path.stem)
        if clip_counter is not None:
            clip_counter[label] = clip_counter.get(label, 0) + n

        return {'clips': n, 'before_dedup': before_dedup, 'after_dedup': after_dedup}

    def _load_frames(self, video_path: Path) -> list:
        cap     = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step    = max(1, round(src_fps / TARGET_FPS))
        frames  = []
        i       = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % step == 0:
                try:
                    f640        = cv2.resize(frame, (640, 640))
                    enhanced, _ = self.enhancer.enhance(f640)
                    f224        = cv2.resize(enhanced, CLIP_SIZE,
                                            interpolation=cv2.INTER_LINEAR)
                    rgb = cv2.cvtColor(f224, cv2.COLOR_BGR2RGB
                                       ).astype(np.float32) / 255.0
                    frames.append(rgb)
                except Exception:
                    pass
            i += 1
        cap.release()
        return frames

    def _make_clips(self, frames: list, n_clips: int) -> list:
        """Uniform segment division with stride-4 sampling and temporal jitter."""
        n      = len(frames)
        seg_sz = n / n_clips
        clips  = []
        for seg_idx in range(n_clips):
            seg_s  = int(seg_idx * seg_sz)
            jitter = random.randint(-TEMPORAL_JITTER, TEMPORAL_JITTER)
            start  = max(0, min(seg_s + jitter, n - 1))
            clip   = self._build_clip_strided(frames, start, n)
            if clip is not None:
                clips.append(clip)
        return clips

    def _build_clip_strided(self, frames: list, start: int, n: int):
        """
        Build (16, 224, 224, 3) by taking every STRIDE-th frame from start.
        Pads with last frame if video ends early. OOM-safe.
        """
        indices = [min(start + k * STRIDE, n - 1) for k in range(CLIP_LEN)]
        try:
            clip = np.stack([frames[i] for i in indices])
        except MemoryError:
            gc.collect()
            return None
        except Exception:
            return None
        return clip if self._valid(clip) else None

    def _valid(self, clip: np.ndarray) -> bool:
        """
        Validate clip shape, pixel range, and motion content.
        OOM-safe: clip.std() was the crash trigger on large videos.
        """
        try:
            if clip.shape != (CLIP_LEN, *CLIP_SIZE, 3):
                return False
            if float(clip.max()) > 1.0 or float(clip.min()) < 0.0:
                return False
            if float(clip.std()) <= 0.01:
                return False
            return True
        except MemoryError:
            gc.collect()
            return False
        except Exception:
            return False

    def _save(self, clips: list, out_dir: Path, stem: str) -> int:
        """Save clips as (16, 224, 224, 3) float32 .npy files. No disk aug."""
        n = 0
        for i, clip in enumerate(clips):
            try:
                np.save(out_dir / f"{stem}_c{i:04d}.npy", clip)
                n += 1
            except Exception as e:
                print(f"    [WARN] Failed to save clip {i}: {e}")
        return n


# ═══════════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_clips(fold: int, split: str = None):
    print(f"\n{'='*70}")
    print(f"  Clip Verification — Fold {fold}  (CLIP_LEN={CLIP_LEN})")
    print(f"{'='*70}")

    expected = {
        'train': 38 * CLIPS_PER_VIDEO['train'],   # 608
        'test' : 12 * CLIPS_PER_VIDEO['test'],    # 96
    }

    for sp in ([split] if split else ['train', 'test']):
        exp = expected[sp]
        print(f"\n  Split: {sp}  (expected ~{exp} clips/class)")
        print(f"  {'Label':10s}  {'Clips':>6}  {'Videos':>7}  "
              f"{'Clips/vid':>9}  {'vs target':>10}  Status")
        print(f"  {'-'*65}")

        for label in LABELS:
            d = CLIPS_DIR / f'fold_{fold}' / sp / label / 'scene'
            if not d.exists():
                print(f"  {label:10s}  NOT FOUND")
                continue

            clip_files = list(d.glob('*.npy'))
            stems      = {re.sub(r'_c\d{4}$', '', p.stem) for p in clip_files}
            n_clips    = len(clip_files)
            n_vids     = len(stems)
            cpv        = n_clips / max(1, n_vids)
            diff       = n_clips - exp

            errors = 0
            for p in clip_files[:20]:
                try:
                    c = np.load(p)
                    if c.shape != (CLIP_LEN, *CLIP_SIZE, 3): errors += 1
                    elif c.max() > 1.0 or c.min() < 0.0:    errors += 1
                    elif c.std() < 0.01:                     errors += 1
                except Exception:
                    errors += 1

            status = "✅" if errors == 0 else f"❌ {errors} errors"
            print(f"  {label:10s}  {n_clips:6d}  {n_vids:7d}  "
                  f"{cpv:9.1f}  {diff:>+10d}  {status}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Process One Fold
# ═══════════════════════════════════════════════════════════════════════════════

def process_fold(fold: int, args, extractor: SceneClipExtractor):
    print(f"\n{'='*70}")
    print(f"  PROCESSING FOLD {fold}  (classes: {LABELS})")
    print(f"{'='*70}")

    fold_split    = load_official_fold(fold)
    splits_to_run = [args.split] if args.split else ['train', 'test']

    for split in splits_to_run:
        per_vid = CLIPS_PER_VIDEO[split]
        print(f"\n{'─'*70}")
        print(f"  fold_{fold}/{split}  (target: {per_vid} clips/video)")
        print(f"{'─'*70}")

        total_stats  = defaultdict(int)
        video_stats  = defaultdict(int)
        clip_counter = {}

        for label in LABELS:
            if args.cls and label != args.cls:
                continue

            video_list = fold_split[split].get(label, [])
            if not video_list:
                print(f"\n[{label}] No videos in fold {fold}/{split}")
                continue
            if args.max_videos:
                video_list = video_list[:args.max_videos]

            print(f"\n[{label}] {len(video_list)} videos | per-video cap={per_vid}")

            for i, (ucf_folder, stem) in enumerate(video_list):
                video_path = None
                for ext in VIDEO_EXTENSIONS:
                    c = VIDEOS_DIR / ucf_folder / f"{stem}{ext}"
                    if c.exists():
                        video_path = c
                        break
                if not video_path:
                    print(f"  [{i+1}/{len(video_list)}] NOT FOUND: {ucf_folder}/{stem}")
                    continue

                try:
                    stats = extractor.extract(
                        video_path   = video_path,
                        label        = label,
                        split        = split,
                        fold         = fold,
                        clip_counter = clip_counter,
                    )
                    if stats.get('skipped'):
                        print(f"  [{i+1}/{len(video_list)}] {stem} → SKIPPED")
                        continue

                    n      = stats['clips']
                    reason = stats.get('reason', '')
                    total_stats[label] += n
                    if n > 0:
                        video_stats[label] += 1

                    dedup_str = (
                        f"  [dedup {stats['before_dedup']}→{stats['after_dedup']}]"
                        if stats.get('before_dedup') else ""
                    )
                    if reason:
                        print(f"  [{i+1}/{len(video_list)}] {stem} → 0 clips ({reason})")
                    else:
                        print(f"  [{i+1}/{len(video_list)}] {stem} "
                              f"→ {n} clips  "
                              f"(total: {clip_counter.get(label,0)})"
                              f"{dedup_str}")

                except Exception as e:
                    import traceback
                    print(f"  [{i+1}/{len(video_list)}] {stem} ERROR: {e}")
                    traceback.print_exc()

        print(f"\n  SUMMARY — fold_{fold}/{split}")
        print(f"  {'Label':10s}  {'Clips':>6}  {'Videos':>7}  {'Clips/vid':>9}")
        for label in LABELS:
            if args.cls and label != args.cls:
                continue
            nc = total_stats.get(label, 0)
            nv = video_stats.get(label, 0)
            print(f"  {label:10s}  {nc:6d}  {nv:7d}  {nc/max(1,nv):9.1f}")

    verify_clips(fold, args.split)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    folds_to_run = list(VALID_FOLDS) if args.all_folds else [args.fold]

    if args.show_split:
        for fold in folds_to_run:
            load_official_fold(fold)
        return

    if args.verify_only:
        for fold in folds_to_run:
            verify_clips(fold, args.split)
        return

    extractor = SceneClipExtractor(device='cuda')

    print(f"\n{'='*70}")
    print(f"  extract_clips v12 — 4 classes (Assault removed)")
    print(f"  Classes  : {LABELS}")
    print(f"  CLIP_LEN : {CLIP_LEN} frames | STRIDE: {STRIDE}")
    print(f"  Train    : {CLIPS_PER_VIDEO['train']} clips/video  "
          f"→ 38 × {CLIPS_PER_VIDEO['train']} = {38*CLIPS_PER_VIDEO['train']}/class")
    print(f"  Test     : {CLIPS_PER_VIDEO['test']}  clips/video  "
          f"→ 12 × {CLIPS_PER_VIDEO['test']}  = {12*CLIPS_PER_VIDEO['test']}/class")
    print(f"  Folds    : {folds_to_run}")
    print(f"{'='*70}\n")

    for fold in folds_to_run:
        process_fold(fold, args, extractor)

    print(f"\n{'='*70}")
    print(f"  ALL DONE — Folds: {folds_to_run}")
    print(f"  Clips at: {CLIPS_DIR}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='extract_clips v12 — 4 classes (Assault removed)'
    )
    fold_grp = parser.add_mutually_exclusive_group()
    fold_grp.add_argument('--fold', type=int, choices=VALID_FOLDS, default=1)
    fold_grp.add_argument('--all-folds', action='store_true')

    parser.add_argument('--split',       choices=['train', 'test'], default=None)
    parser.add_argument('--class',       dest='cls', default=None, choices=LABELS)
    parser.add_argument('--max-videos',  type=int, default=None)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--show-split',  action='store_true')

    args = parser.parse_args()
    main(args)