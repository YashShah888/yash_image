"""Build ai-toolkit configs from the validator-compatible G.O.D templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from image_recipe import recipe_table, task_shape

CACHE_ROOT = Path("/cache")
CONFIG_TEMPLATE_ROOT = Path("/workspace/core/training_templates")
TEMPLATE_BY_MODEL_TYPE = {
    "flux": "base_diffusion_flux.yaml",
    "z-image": "base_diffusion_zimage.yaml",
    "qwen-image": "base_diffusion_qwen_image.yaml",
    "ideogram4": "base_diffusion_ideogram4.yaml",
    "krea2": "base_diffusion_krea2.yaml",
}
IDEOGRAM4_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-8B-Instruct"
KREA2_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-4B-Instruct"
_TEMPLATES_WITH_CAPTION_DROPOUT = {"flux", "ideogram4", "krea2"}


def cached_model_path(model: str) -> Path:
    return CACHE_ROOT / "models" / model.replace("/", "--")


def _first_process(config: dict[str, Any]) -> dict[str, Any]:
    body = config.setdefault("config", {})
    processes = body.setdefault("process", [{}])
    if not processes:
        processes.append({})
    return processes[0]


def _model_specific_paths(model_type: str, model_config: dict[str, Any], model_path: Path) -> None:
    if model_type == "ideogram4":
        if not IDEOGRAM4_TEXT_ENCODER_CACHE.exists():
            raise FileNotFoundError(
                f"expected cached Ideogram 4 text encoder at {IDEOGRAM4_TEXT_ENCODER_CACHE}"
            )
        model_config.setdefault("model_kwargs", {})["text_encoder_path"] = str(
            IDEOGRAM4_TEXT_ENCODER_CACHE
        )
    elif model_type == "krea2":
        if not KREA2_TEXT_ENCODER_CACHE.exists():
            raise FileNotFoundError(
                f"expected cached Krea 2 text encoder at {KREA2_TEXT_ENCODER_CACHE}"
            )
        vae_path = model_path / "vae"
        if not vae_path.exists():
            raise FileNotFoundError(f"expected cached Krea 2 VAE at {vae_path}")
        kwargs = model_config.setdefault("model_kwargs", {})
        kwargs["text_encoder_path"] = str(KREA2_TEXT_ENCODER_CACHE)
        # Preserve the path convention used by the current G.O.D entrypoint.
        kwargs["vae_path"] = str(model_path)


def _set_if_present(mapping: dict[str, Any], key: str, value: Any) -> bool:
    if key in mapping:
        mapping[key] = value
        return True
    return False


def build_config(
    task_id: str,
    model: str,
    model_type: str,
    expected_repo_name: str,
    dataset_path: Path,
    checkpoints_root: Path,
    trigger_word: str | None,
    hours_to_complete: float = 1.5,
    **_: Any,
) -> tuple[dict[str, Any], recipe_table.Recipe, task_shape.ShapeResult]:
    if model_type not in TEMPLATE_BY_MODEL_TYPE:
        raise ValueError(f"unsupported model type: {model_type}")
    template_path = CONFIG_TEMPLATE_ROOT / TEMPLATE_BY_MODEL_TYPE[model_type]
    if not template_path.exists():
        raise FileNotFoundError(f"missing ai-toolkit template: {template_path}")
    with template_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"invalid template root in {template_path}")

    config_body = config.setdefault("config", {})
    config_body["name"] = expected_repo_name
    process = _first_process(config)
    process["training_folder"] = str(checkpoints_root)
    process["trigger_word"] = trigger_word

    datasets = process.setdefault("datasets", [{}])
    if not datasets:
        datasets.append({})
    dataset = datasets[0]
    dataset["folder_path"] = str(dataset_path)

    model_path = cached_model_path(model)
    if not model_path.exists():
        raise FileNotFoundError(f"expected cached base model at {model_path}")
    model_config = process.setdefault("model", {})
    model_config["name_or_path"] = str(model_path)
    _model_specific_paths(model_type, model_config, model_path)

    captions = task_shape.read_captions(dataset_path)
    images = task_shape.list_images(dataset_path)
    shape = task_shape.classify(captions, images)
    recipe = recipe_table.build_recipe(
        model_type=model_type,
        is_subject=shape.is_subject,
        category=shape.category,
        category_confident=shape.confident_from_text or shape.image_signal_used,
        n_images=len(images),
        template_supports_caption_dropout=model_type in _TEMPLATES_WITH_CAPTION_DROPOUT,
        hours_to_complete=hours_to_complete,
    )

    network = process.setdefault("network", {})
    network["linear"] = recipe.rank
    network["linear_alpha"] = recipe.alpha

    train = process.setdefault("train", {})
    train["steps"] = recipe.steps
    train["lr"] = recipe.learning_rate
    # Preview sampling consumes meaningful H100 time and the old pixel-MSE
    # selector was not semantically valid.  Put all budget into optimization.
    train["disable_sampling"] = True
    train["skip_first_sample"] = True

    if recipe.caption_dropout_rate is not None:
        dataset["caption_dropout_rate"] = recipe.caption_dropout_rate

    # Ideogram 4 and Krea 2 expose an explicit content-vs-style training mode.
    # Use the detected task shape instead of the template's one-size-balanced
    # default; keep graphic layouts balanced because they require both form and
    # visual treatment.
    if "content_or_style" in train:
        if shape.category in {"person", "product"}:
            train["content_or_style"] = "content"
        elif shape.category == "style":
            train["content_or_style"] = "style"
        else:
            train["content_or_style"] = "balanced"

    save = process.setdefault("save", {})
    save["save_every"] = recipe.save_every
    save["max_step_saves_to_keep"] = recipe.max_saves
    # Some template versions use this spelling instead.  Change it only when
    # already present, avoiding unknown schema keys.
    _set_if_present(save, "max_saves_to_keep", recipe.max_saves)

    # Do not silently train text encoders.  All current templates are intended
    # for adapter training; if the field exists, preserve that invariant.
    _set_if_present(train, "train_text_encoder", False)
    return config, recipe, shape


def save_config(config: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
