#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_recipe import caption
from image_recipe import checkpoint_select
from image_recipe import config_builder
from image_recipe import dataset_prep


CACHE_ROOT = Path("/cache")
DATASET_ZIP_ROOT = CACHE_ROOT / "datasets"
IMAGE_DATASET_ROOT = Path("/dataset/images")
HOLDOUT_ROOT = Path("/dataset/holdout")
CONFIG_OUTPUT_ROOT = Path("/dataset/configs")
CHECKPOINTS_ROOT = Path("/app/checkpoints")
AI_TOOLKIT_ROOT = Path("/app/ai-toolkit")

# Fraction (and cap) of hours_to_complete reserved so the training subprocess
# is stopped early enough to leave time for checkpoint selection and the
# platform's own upload step. Selection itself only reads already-generated
# sample images (cheap), so this reserve is small.
RESERVE_FRACTION = 0.08
RESERVE_MIN_MINUTES = 2
RESERVE_MAX_MINUTES = 12


def prepare_dataset(task_id: str) -> Path:
    zip_path = DATASET_ZIP_ROOT / f"{task_id}_tourn.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected cached image dataset zip at {zip_path}")

    if IMAGE_DATASET_ROOT.exists():
        shutil.rmtree(IMAGE_DATASET_ROOT)
    IMAGE_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    extract_root = Path("/dataset/extracted") / task_id
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    dataset_prep.safe_extract(str(zip_path), str(extract_root))

    supported_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".txt"}
    copied = 0
    for source in extract_root.rglob("*"):
        if source.is_file() and source.suffix.lower() in supported_suffixes:
            destination = IMAGE_DATASET_ROOT / source.name
            if destination.exists():
                destination = IMAGE_DATASET_ROOT / f"{source.parent.name}_{source.name}"
            shutil.copy2(source, destination)
            copied += 1

    if copied == 0:
        raise RuntimeError(f"No supported image/caption files found in {zip_path}")
    return IMAGE_DATASET_ROOT


def training_timeout_seconds(hours_to_complete: float) -> float:
    total_minutes = max(1.0, hours_to_complete * 60.0)
    reserve = min(RESERVE_MAX_MINUTES, max(RESERVE_MIN_MINUTES, total_minutes * RESERVE_FRACTION))
    return max(60.0, (total_minutes - reserve) * 60.0)


def run_training(config_path: Path, timeout_seconds: float) -> None:
    print(f"Starting training with config: {config_path} (timeout={timeout_seconds:.0f}s)", flush=True)
    process = subprocess.Popen(
        ["python3", "run.py", str(config_path)],
        cwd=AI_TOOLKIT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    start = time.monotonic()
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if time.monotonic() - start > timeout_seconds:
                print("[trainer] time-budget reserve reached; stopping training", flush=True)
                process.terminate()
                break
        return_code = process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        print("[trainer] training did not exit after terminate; killing", flush=True)
        process.kill()
        return_code = process.wait()

    if return_code not in (0, None) and return_code < 0:
        # Negative return code means the process was terminated by a signal
        # (our own reserve-boundary terminate/kill) -- treat as an intentional
        # stop, not a failure, as long as at least one checkpoint exists.
        print(f"[trainer] training subprocess stopped by signal {return_code} (reserve boundary)", flush=True)
        return
    if return_code != 0:
        raise RuntimeError(f"Training subprocess failed with exit code {return_code}")
    print("Training subprocess completed successfully.", flush=True)


def main() -> None:
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument("--model-type", required=True, choices=sorted(config_builder.TEMPLATE_BY_MODEL_TYPE))
    parser.add_argument("--expected-repo-name", required=True)
    parser.add_argument("--hours-to-complete", required=True, type=float)
    parser.add_argument("--trigger-word")
    args = parser.parse_args()

    dataset_dir = prepare_dataset(args.task_id)

    dedup = dataset_prep.perceptual_dedup(str(dataset_dir))
    if dedup.n_removed:
        print(f"[dedup] removed {dedup.n_removed} near-duplicate images "
              f"({dedup.n_before}->{dedup.n_after}, dup_rate={dedup.dup_rate})", flush=True)
    if dedup.note:
        print(f"[dedup] {dedup.note}", flush=True)

    holdout_dir = HOLDOUT_ROOT / args.task_id
    holdout = dataset_prep.holdout_split(str(dataset_dir), str(holdout_dir))
    if holdout.note:
        print(f"[holdout] {holdout.note}", flush=True)
    else:
        print(f"[holdout] held out {len(holdout.held_image_paths)} images for checkpoint selection", flush=True)

    if args.trigger_word:
        caption.apply_trigger_to_dir(str(dataset_dir), args.trigger_word)
        caption.apply_trigger_to_dir(str(holdout_dir), args.trigger_word)

    # Read captions in the same order as holdout.held_image_paths (not a
    # re-sorted directory listing) so the Nth prompt handed to ai-toolkit's
    # sampler lines up with the Nth held-out image checkpoint_select compares
    # generated samples against.
    held_out_captions = []
    for cap_path in holdout.held_caption_paths:
        try:
            with open(cap_path, encoding="utf-8") as fh:
                held_out_captions.append(fh.read().strip())
        except OSError:
            held_out_captions.append("")

    checkpoints_dir = CHECKPOINTS_ROOT / args.task_id
    config, recipe, shape = config_builder.build_config(
        task_id=args.task_id,
        model=args.model,
        model_type=args.model_type,
        expected_repo_name=args.expected_repo_name,
        dataset_path=dataset_dir,
        checkpoints_root=checkpoints_dir,
        trigger_word=args.trigger_word,
        held_out_captions=held_out_captions,
        enable_sampling=bool(holdout.held_image_paths),
    )
    print(f"[recipe] shape={shape.category} subject={shape.is_subject} "
          f"confident={shape.confident_from_text} image_signal={shape.image_signal_used} "
          f"rank={recipe.rank} step_ceiling={recipe.step_ceiling} "
          f"caption_dropout={recipe.caption_dropout_rate}", flush=True)
    for note in shape.notes + recipe.notes:
        print(f"[recipe] {note}", flush=True)

    config_path = CONFIG_OUTPUT_ROOT / f"{args.task_id}.yaml"
    config_builder.save_config(config, config_path)
    print(f"Created ai-toolkit config at {config_path}", flush=True)

    timeout_seconds = training_timeout_seconds(args.hours_to_complete)
    run_training(config_path, timeout_seconds)

    output_dir = checkpoints_dir / args.expected_repo_name
    result = checkpoint_select.select_best(str(output_dir), holdout.held_image_paths)
    if result:
        print(f"[checkpoint_select] chose step {result.chosen_step} "
              f"(scores={result.scores}, removed_steps={result.removed_steps})", flush=True)
    else:
        print("[checkpoint_select] no selection made; leaving trainer's own output as-is", flush=True)


if __name__ == "__main__":
    main()
