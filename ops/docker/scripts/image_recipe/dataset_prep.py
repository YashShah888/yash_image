"""
Dataset preparation: safe extraction, near-duplicate removal, and a held-out
validation split.

The dataset zip is validator-supplied infrastructure, not arbitrary user
upload, but it costs nothing to extract it defensively rather than trust
zipfile.extractall blindly -- a hostile or corrupted archive shouldn't be
able to write outside the target directory or exhaust disk via a zip bomb.

Near-dup removal and the holdout split are guarded to never raise -- a
dataset-prep failure must degrade to "use the full dataset as-is", not kill
the training run.
"""

from __future__ import annotations

import os
import random
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MAX_MEMBER_BYTES = 2_000_000_000
MAX_TOTAL_BYTES = 10_000_000_000


def safe_extract(zip_path: str, destination: str) -> None:
    """Extract `zip_path` into `destination`, rejecting members that would
    escape the destination directory (path traversal / absolute paths),
    symlinks, encrypted entries, or an archive that expands past a sane size
    ceiling (zip-bomb guard)."""
    dest_root = Path(destination).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    total_uncompressed = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (dest_root / member.filename).resolve()
            if target != dest_root and dest_root not in target.parents:
                raise ValueError(f"Unsafe path in dataset archive: {member.filename!r}")
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ValueError(f"Symlinks are not allowed in the dataset archive: {member.filename!r}")
            if member.flag_bits & 0x1:
                raise ValueError(f"Encrypted dataset members are unsupported: {member.filename!r}")
            if member.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"Dataset member is unexpectedly large: {member.filename!r}")
            total_uncompressed += member.file_size
            if total_uncompressed > MAX_TOTAL_BYTES:
                raise ValueError("Dataset archive expands beyond the safety limit")
        archive.extractall(dest_root)


def _caption_path(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + ".txt"


def _list_pairs(image_dir: str) -> list[str]:
    if not os.path.isdir(image_dir):
        return []
    return [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
    ]


@dataclass
class DedupResult:
    n_before: int
    n_after: int
    n_removed: int
    dup_rate: float
    note: str | None = None


def _thumb_signature(img) -> "list[float]":
    """Cheap per-channel-mean thumbnail signature used as a second, phash-
    independent duplicate signal (see perceptual_dedup)."""
    small = img.convert("RGB").resize((8, 8))
    pixels = list(small.getdata())
    n = len(pixels)
    return [sum(c[i] for c in pixels) / n for i in range(3)]


def _thumb_distance(a: "list[float]", b: "list[float]") -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _sharpness(img) -> float:
    """Cheap gradient-magnitude proxy for image sharpness (no numpy/scipy
    dependency): sum of absolute horizontal + vertical pixel deltas on a
    small grayscale thumbnail. Used to pick the best representative out of a
    near-duplicate cluster instead of an arbitrary one."""
    small = img.convert("L").resize((64, 64))
    pixels = list(small.getdata())
    total = 0
    for y in range(64):
        row = pixels[y * 64 : (y + 1) * 64]
        total += sum(abs(row[x + 1] - row[x]) for x in range(63))
        if y > 0:
            prev = pixels[(y - 1) * 64 : y * 64]
            total += sum(abs(row[x] - prev[x]) for x in range(64))
    return total


def perceptual_dedup(
    image_dir: str,
    hamming_threshold: int = 4,
    thumb_threshold: float = 12.0,
) -> DedupResult:
    """Remove near-duplicate images (and their caption files) in-place,
    keeping the sharpest representative of each near-duplicate cluster
    (not just whichever one was encountered first).

    A pair only counts as a duplicate when BOTH a phash structural match
    (hamming_threshold) AND a coarse per-channel colour-thumbnail match
    (thumb_threshold) agree. phash alone is nearly blind to flat, low-texture
    images (logos, social/design graphics -- exactly categories this subnet
    covers) and will call two genuinely different flat-colour images
    duplicates; requiring the colour signal too avoids that failure mode
    while still catching true near-duplicates, which are close on both axes.
    """
    paths = _list_pairs(image_dir)
    n_before = len(paths)
    if n_before < 3:
        return DedupResult(n_before, n_before, 0, 0.0, note="too few images to dedup")

    try:
        import imagehash
        from PIL import Image
    except ImportError:
        return DedupResult(n_before, n_before, 0, 0.0, note="PIL/imagehash unavailable")

    signatures: list[tuple[str, object, list[float], float]] = []
    for path in paths:
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                signatures.append((path, imagehash.phash(rgb), _thumb_signature(rgb), _sharpness(rgb)))
        except Exception:
            continue

    # Union-find clustering: group every image with every other image it's a
    # near-duplicate of (not just adjacent ones), then keep only the sharpest
    # member of each cluster.
    n = len(signatures)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            _, hi, thumb_i, _ = signatures[i]
            _, hj, thumb_j, _ = signatures[j]
            if (hi - hj) <= hamming_threshold and _thumb_distance(thumb_i, thumb_j) <= thumb_threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    removed = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        best = max(members, key=lambda idx: signatures[idx][3])
        for idx in members:
            if idx == best:
                continue
            path = signatures[idx][0]
            try:
                os.remove(path)
                cap = _caption_path(path)
                if os.path.exists(cap):
                    os.remove(cap)
                removed += 1
            except OSError:
                pass

    n_after = n_before - removed
    dup_rate = round(removed / n_before, 4) if n_before else 0.0
    return DedupResult(n_before, n_after, removed, dup_rate)


@dataclass
class HoldoutResult:
    held_image_paths: list[str]
    held_caption_paths: list[str]
    note: str | None = None


def holdout_split(
    image_dir: str,
    holdout_dir: str,
    fraction: float = 0.15,
    min_holdout: int = 1,
    max_holdout: int = 4,
    min_remaining: int = 4,
    seed: int = 0,
) -> HoldoutResult:
    """Move a small slice of (image, caption) pairs out of `image_dir` into
    `holdout_dir`, for later use as an internal validation set. Skips
    entirely (returns an empty result) when the dataset is too small to
    spare any images without starving training."""
    paths = _list_pairs(image_dir)
    n = len(paths)
    if n < min_remaining + min_holdout:
        return HoldoutResult([], [], note=f"dataset too small to hold out (n={n})")

    n_holdout = max(min_holdout, min(max_holdout, round(n * fraction)))
    n_holdout = min(n_holdout, n - min_remaining)
    if n_holdout <= 0:
        return HoldoutResult([], [], note="no room for a holdout slice")

    rng = random.Random(seed)
    # Only offer images that have a caption file as holdout candidates, so
    # the returned image/caption lists stay strictly index-aligned -- callers
    # rely on that alignment to match generated preview images (by prompt
    # position) back to the real held-out image they were sampled against.
    candidates = [p for p in paths if os.path.exists(_caption_path(p))]
    if len(candidates) < n_holdout:
        n_holdout = len(candidates)
    if n_holdout <= 0:
        return HoldoutResult([], [], note="no captioned images available to hold out")
    chosen = rng.sample(candidates, n_holdout)

    os.makedirs(holdout_dir, exist_ok=True)
    held_images, held_captions = [], []
    for src in chosen:
        cap_src = _caption_path(src)
        dst = os.path.join(holdout_dir, os.path.basename(src))
        cap_dst = _caption_path(dst)
        try:
            shutil.move(src, dst)
            shutil.move(cap_src, cap_dst)
        except OSError:
            continue
        held_images.append(dst)
        held_captions.append(cap_dst)

    return HoldoutResult(held_images, held_captions)
