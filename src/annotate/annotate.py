"""Interactive timestamp annotator.

Scrub a recorded gameplay video and press an ability key at the exact moment a
spell is cast. Events are written to ``data/annotations/<video_stem>.json`` and
re-loaded on the next run so you can resume.

Run::

    python -m src.annotate.annotate --config configs/ezreal.yaml \
        --video data/raw_videos/ezreal_game1.mp4

This opens an OpenCV window; controls are printed on launch and shown on screen.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from src.common.annotations import Annotation, Event, annotation_path_for
from src.common.config import Config

# Control keys (reserved; ability letter-keys that collide fall back to digits).
KEY_QUIT = 27          # ESC
KEY_SPACE = 32         # play / pause
RESERVED = {KEY_SPACE, ord(","), ord("."), ord("["), ord("]"),
            ord("u"), ord("x"), ord("s"), ord("h"), KEY_QUIT}

SEEK_SECONDS = 1.0
DELETE_TOLERANCE_SEC = 0.5
MAX_DISPLAY_WIDTH = 1280


def build_keymap(abilities: List[str]) -> Dict[int, str]:
    """Map keycodes to ability names.

    Each ability gets a digit key (1..N) and, when free, its lowercased first
    letter (natural for Q/W/E/R).
    """
    keymap: Dict[int, str] = {}
    for i, name in enumerate(abilities):
        digit = ord(str(i + 1)) if i < 9 else None
        if digit is not None and digit not in keymap:
            keymap[digit] = name
        letter = ord(name[0].lower())
        if letter not in RESERVED and letter not in keymap:
            keymap[letter] = name
    return keymap


def keymap_help(keymap: Dict[int, str]) -> str:
    parts = []
    for code, name in keymap.items():
        parts.append(f"'{chr(code)}'={name}")
    return ", ".join(parts)


def _counts(ann: Annotation, abilities: List[str]) -> str:
    counts = {a: 0 for a in abilities}
    for e in ann.events:
        if e.ability in counts:
            counts[e.ability] += 1
    return " ".join(f"{a}:{counts[a]}" for a in abilities) + f" | total {len(ann.events)}"


def _nearby(ann: Annotation, t: float, window: float = 2.0) -> List[Event]:
    return sorted(
        [e for e in ann.events if abs(e.time - t) <= window],
        key=lambda e: e.time,
    )


def _draw_overlay(
    frame,
    *,
    pos: int,
    total_frames: int,
    fps: float,
    playing: bool,
    ann: Annotation,
    abilities: List[str],
    keymap: Dict[int, str],
    flash: Optional[str],
):
    img = frame.copy()
    h, w = img.shape[:2]
    scale = 1.0
    if w > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    t = pos / fps if fps else 0.0
    total_t = total_frames / fps if fps else 0.0

    def put(text, y, color=(255, 255, 255), size=0.6):
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    state = "PLAYING" if playing else "PAUSED"
    put(f"{state}  t={t:6.2f}/{total_t:6.2f}s  frame={pos}/{total_frames}", 24,
        (0, 255, 0) if playing else (0, 200, 255))
    put(_counts(ann, abilities), 48, (255, 220, 0))

    near = _nearby(ann, t)
    near_txt = "near: " + (", ".join(f"{e.ability}@{e.time:.2f}" for e in near) if near else "-")
    put(near_txt, 72, (200, 200, 255))

    if flash:
        put(flash, 96, (0, 255, 255), size=0.7)

    controls = ("SPACE play/pause  ,/. frame  [/] 1s  u undo  x del-near  s save  ESC quit  | "
                + keymap_help(keymap))
    put(controls, h - 12, (180, 180, 180), size=0.5)
    return img


def run(config_path: str, video_path: str, annotations_dir: str) -> None:
    cfg = Config.load(config_path)
    abilities = cfg.ability_classes
    keymap = build_keymap(abilities)

    video_path = str(Path(video_path))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ann_path = annotation_path_for(video_path, annotations_dir)
    ann = Annotation.load_or_new(ann_path, video=Path(video_path).name, fps=fps)
    print(f"Video: {video_path}  ({total_frames} frames @ {fps:.2f} fps)")
    print(f"Annotations: {ann_path}  ({len(ann.events)} existing events)")
    print(f"Ability keys: {keymap_help(keymap)}")
    print("Controls: SPACE play/pause | , . step frame | [ ] seek 1s | u undo | x delete-near | s save | ESC quit")

    win = f"annotate: {Path(video_path).name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    pos = 0
    playing = False
    resync = False
    frame = None
    flash: Optional[str] = None
    flash_ticks = 0
    dirty = True

    def clamp(p: int) -> int:
        return max(0, min(p, max(total_frames - 1, 0)))

    def grab(p: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
        ok, f = cap.read()
        return f if ok else None

    while True:
        if playing:
            if resync:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos + 1)
                resync = False
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
                pos += 1
                if pos >= total_frames - 1:
                    playing = False
            else:
                playing = False
        elif dirty:
            f = grab(pos)
            if f is not None:
                frame = f
            dirty = False

        if frame is not None:
            if flash_ticks > 0:
                flash_ticks -= 1
            else:
                flash = None
            shown = _draw_overlay(
                frame, pos=pos, total_frames=total_frames, fps=fps, playing=playing,
                ann=ann, abilities=abilities, keymap=keymap, flash=flash,
            )
            cv2.imshow(win, shown)

        delay = max(1, int(1000 / fps)) if playing else 20
        key = cv2.waitKey(delay) & 0xFF
        if key == 255:  # no key
            # Window closed via the title bar?
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        if key == KEY_QUIT:
            break
        elif key == KEY_SPACE:
            playing = not playing
            if playing:
                resync = True
            else:
                dirty = True
        elif key == ord(","):
            playing = False
            pos = clamp(pos - 1)
            dirty = True
        elif key == ord("."):
            playing = False
            pos = clamp(pos + 1)
            dirty = True
        elif key == ord("["):
            playing = False
            pos = clamp(pos - int(SEEK_SECONDS * fps))
            dirty = True
        elif key == ord("]"):
            playing = False
            pos = clamp(pos + int(SEEK_SECONDS * fps))
            dirty = True
        elif key == ord("s"):
            ann.save(ann_path)
            flash, flash_ticks = f"saved -> {ann_path.name} ({len(ann.events)} events)", 40
            print(flash)
        elif key == ord("u"):
            if ann.events:
                removed = ann.events.pop()
                flash, flash_ticks = f"undo {removed.ability}@{removed.time:.2f}", 40
            else:
                flash, flash_ticks = "nothing to undo", 40
        elif key == ord("x"):
            t = pos / fps if fps else 0.0
            candidates = [(abs(e.time - t), idx) for idx, e in enumerate(ann.events)
                          if abs(e.time - t) <= DELETE_TOLERANCE_SEC]
            if candidates:
                _, idx = min(candidates)
                removed = ann.events.pop(idx)
                flash, flash_ticks = f"deleted {removed.ability}@{removed.time:.2f}", 40
            else:
                flash, flash_ticks = "no event within tolerance", 40
        elif key in keymap:
            t = pos / fps if fps else 0.0
            ann.events.append(Event(time=t, frame=pos, ability=keymap[key]))
            flash, flash_ticks = f"+ {keymap[key]} @ {t:.2f}s (frame {pos})", 40

    ann.save(ann_path)
    print(f"Saved {len(ann.events)} events -> {ann_path}")
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Timestamp annotator for LoL ability casts.")
    p.add_argument("--config", required=True, help="Path to champion config YAML")
    p.add_argument("--video", required=True, help="Path to a video in data/raw_videos")
    p.add_argument("--annotations-dir", default="data/annotations",
                   help="Where to read/write annotation JSON")
    args = p.parse_args()
    run(args.config, args.video, args.annotations_dir)


if __name__ == "__main__":
    main()
