"""
No-GPU tests for the adaptive image-tournament training recipe
(ops/docker/scripts/image_recipe/). These modules live outside the normal
package tree because they're COPYed standalone into the training container
image (see ops/docker/standalone-image-toolkit-trainer.dockerfile), so we add
their directory to sys.path here rather than importing them as part of the
`core`/`trainer` packages.

Run with: uv run --extra dev pytest tests/trainer/image_recipe -v
"""

import os
import sys
import tempfile
import zipfile

import pytest


_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "docker", "scripts"))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from image_recipe import checkpoint_select  # noqa: E402
from image_recipe import dataset_prep  # noqa: E402
from image_recipe import recipe_table  # noqa: E402
from image_recipe import task_shape  # noqa: E402


CAPTION_CASES = {
    "person": "A smiling woman wearing a red jacket, portrait, studio lighting",
    "product": "Product shot of a sneaker isolated on white background, e-commerce catalog",
    "logo": "A minimalist wordmark logo, brandmark, monogram in navy blue",
    "social": "Instagram story template with bold headline and call to action",
    "design": "App screen mockup, dashboard UI with buttons and icons",
    "style": "Digital art in a painterly art style, vibrant palette, brushwork",
}


@pytest.mark.parametrize("expected,caption", CAPTION_CASES.items())
def test_classify_from_captions(expected, caption):
    category, confident = task_shape.classify_from_captions([caption])
    assert category == expected
    assert confident


def test_classify_from_captions_no_signal_falls_back():
    category, confident = task_shape.classify_from_captions(["", "   "])
    assert category == task_shape.FALLBACK_CATEGORY
    assert not confident


@pytest.mark.parametrize("category,expected_subject", [
    ("person", True),
    ("product", True),
    ("style", False),
    ("logo", False),
    ("social", False),
    ("design", False),
    ("unknown-category", False),
])
def test_resolve_shape(category, expected_subject):
    assert task_shape.resolve_shape(category) is expected_subject


@pytest.mark.parametrize("model_type", ["flux", "z-image", "qwen-image", "ideogram4", "krea2"])
def test_recipe_table_covers_all_five_model_types(model_type):
    """The champion repo we studied only handles sdxl/flux/qwen-image/z-image
    -- ideogram4 and krea2 are unsupported there. This is the gap our recipe
    table (and config_builder) closes; every model type must produce a
    usable recipe."""
    recipe = recipe_table.build_recipe(
        model_type=model_type, is_subject=True, category="person",
        category_confident=True, n_images=12, template_supports_caption_dropout=True,
    )
    assert recipe.rank > 0
    assert recipe.step_ceiling > 0


def test_recipe_table_subject_gets_tighter_ceiling_than_style():
    subject = recipe_table.build_recipe(
        "flux", is_subject=True, category="person", category_confident=True,
        n_images=20, template_supports_caption_dropout=True,
    )
    style = recipe_table.build_recipe(
        "flux", is_subject=False, category="style", category_confident=True,
        n_images=20, template_supports_caption_dropout=True,
    )
    assert subject.step_ceiling < style.step_ceiling


def test_recipe_table_caption_dropout_only_when_template_supports_it():
    with_support = recipe_table.build_recipe(
        "flux", is_subject=False, category="style", category_confident=True,
        n_images=20, template_supports_caption_dropout=True,
    )
    without_support = recipe_table.build_recipe(
        "z-image", is_subject=False, category="style", category_confident=True,
        n_images=20, template_supports_caption_dropout=False,
    )
    assert with_support.caption_dropout_rate is not None
    assert without_support.caption_dropout_rate is None


def test_dataset_prep_holdout_split_keeps_images_and_captions_aligned():
    with tempfile.TemporaryDirectory() as tmp:
        image_dir = os.path.join(tmp, "images")
        holdout_dir = os.path.join(tmp, "holdout")
        os.makedirs(image_dir)
        for i in range(10):
            open(os.path.join(image_dir, f"img{i}.jpg"), "wb").close()
            with open(os.path.join(image_dir, f"img{i}.txt"), "w") as fh:
                fh.write(f"caption {i}")

        result = dataset_prep.holdout_split(image_dir, holdout_dir, min_remaining=4, max_holdout=3)
        assert result.note is None
        assert len(result.held_image_paths) == len(result.held_caption_paths) > 0
        for img_path, cap_path in zip(result.held_image_paths, result.held_caption_paths):
            assert os.path.exists(img_path)
            assert os.path.exists(cap_path)
            assert os.path.dirname(img_path) == holdout_dir


def test_dataset_prep_holdout_split_skips_when_dataset_too_small():
    with tempfile.TemporaryDirectory() as tmp:
        image_dir = os.path.join(tmp, "images")
        os.makedirs(image_dir)
        for i in range(3):
            open(os.path.join(image_dir, f"img{i}.jpg"), "wb").close()
            open(os.path.join(image_dir, f"img{i}.txt"), "w").close()

        result = dataset_prep.holdout_split(image_dir, os.path.join(tmp, "holdout"), min_remaining=4)
        assert result.held_image_paths == []
        assert result.note is not None


def test_dedup_distinguishes_flat_color_images_from_true_near_duplicates():
    """Regression test for a real bug found during development: phash alone
    is nearly blind to flat/low-texture images (exactly the logo/social/
    design categories this subnet covers) and would call distinct flat-color
    images duplicates. dataset_prep.perceptual_dedup requires a second,
    colour-thumbnail signal to agree before removing anything."""
    pytest.importorskip("PIL")
    pytest.importorskip("imagehash")
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        # Six distinct flat colours -- must all survive.
        for i in range(6):
            color = (i * 40 % 255, (i * 80) % 255, (i * 120) % 255)
            Image.new("RGB", (32, 32), color).save(os.path.join(tmp, f"img{i}.jpg"))
            open(os.path.join(tmp, f"img{i}.txt"), "w").write(f"caption {i}")

        result = dataset_prep.perceptual_dedup(tmp)
        assert result.n_removed == 0, "distinct flat-colour images were wrongly treated as duplicates"


def test_safe_extract_rejects_path_traversal():
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "evil.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../../etc/evil.txt", "pwned")

        with pytest.raises(ValueError):
            dataset_prep.safe_extract(zip_path, os.path.join(tmp, "out"))


def test_safe_extract_allows_a_normal_archive():
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "ok.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("img1.jpg", b"fakejpegbytes")
            zf.writestr("img1.txt", "a caption")

        dest = os.path.join(tmp, "out")
        dataset_prep.safe_extract(zip_path, dest)
        assert sorted(os.listdir(dest)) == ["img1.jpg", "img1.txt"]


def test_dedup_keeps_the_sharper_image_in_a_near_duplicate_cluster():
    """perceptual_dedup should keep the best-quality (sharpest) representative
    of a near-duplicate cluster, not an arbitrary/first-encountered one."""
    pytest.importorskip("PIL")
    pytest.importorskip("imagehash")
    from PIL import Image
    from PIL import ImageFilter

    with tempfile.TemporaryDirectory() as tmp:
        sharp = Image.new("RGB", (64, 64))
        px = sharp.load()
        for x in range(64):
            for y in range(64):
                v = 255 if (x // 4 + y // 4) % 2 == 0 else 0
                px[x, y] = (v, v, v)
        sharp.save(os.path.join(tmp, "sharp.png"))
        open(os.path.join(tmp, "sharp.txt"), "w").write("sharp version")

        blurry = sharp.filter(ImageFilter.GaussianBlur(radius=2))
        blurry.save(os.path.join(tmp, "blurry.png"))
        open(os.path.join(tmp, "blurry.txt"), "w").write("blurry version")

        for i in range(3):
            Image.new("RGB", (64, 64), (i * 70, 255 - i * 70, 100)).save(os.path.join(tmp, f"filler{i}.png"))
            open(os.path.join(tmp, f"filler{i}.txt"), "w").write(f"filler {i}")

        result = dataset_prep.perceptual_dedup(tmp, hamming_threshold=8, thumb_threshold=20.0)
        remaining = os.listdir(tmp)
        assert result.n_removed == 1
        assert "sharp.png" in remaining
        assert "blurry.png" not in remaining

    with tempfile.TemporaryDirectory() as tmp:
        base = Image.new("RGB", (32, 32), (120, 60, 200))
        base.save(os.path.join(tmp, "orig.jpg"))
        open(os.path.join(tmp, "orig.txt"), "w").write("original")
        # Near-identical copy (true duplicate).
        base.save(os.path.join(tmp, "near_dup.jpg"))
        open(os.path.join(tmp, "near_dup.txt"), "w").write("near dup")
        # Well-separated distractors.
        for i, color in enumerate([(0, 200, 50), (255, 255, 0), (10, 10, 10)]):
            Image.new("RGB", (32, 32), color).save(os.path.join(tmp, f"other{i}.jpg"))
            open(os.path.join(tmp, f"other{i}.txt"), "w").write(f"other {i}")

        result = dataset_prep.perceptual_dedup(tmp)
        assert result.n_removed == 1


def test_checkpoint_select_no_op_when_only_one_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "repo_000000250.safetensors"), "w").close()
        assert checkpoint_select.select_best(tmp, ["/nonexistent.jpg"]) is None


def test_checkpoint_select_no_op_without_samples():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "repo_000000250.safetensors"), "w").close()
        open(os.path.join(tmp, "repo_000000500.safetensors"), "w").close()
        assert checkpoint_select.select_best(tmp, ["/nonexistent.jpg"]) is None


def test_checkpoint_select_picks_the_closer_checkpoint_and_prunes_the_other():
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        samples_dir = os.path.join(tmp, "samples")
        os.makedirs(samples_dir)

        holdout_path = os.path.join(tmp, "holdout1.jpg")
        Image.new("RGB", (64, 64), (255, 0, 0)).save(holdout_path)

        # Checkpoint 250's sample is close to the real held-out image.
        Image.new("RGB", (64, 64), (240, 10, 10)).save(os.path.join(samples_dir, "sample_000000250_0.jpg"))
        open(os.path.join(tmp, "repo_000000250.safetensors"), "w").close()

        # Checkpoint 500 has drifted away (simulates overfitting).
        Image.new("RGB", (64, 64), (10, 10, 240)).save(os.path.join(samples_dir, "sample_000000500_0.jpg"))
        open(os.path.join(tmp, "repo_000000500.safetensors"), "w").close()

        result = checkpoint_select.select_best(tmp, [holdout_path])

        assert result is not None
        assert result.chosen_step == 250
        remaining = sorted(os.listdir(tmp))
        assert "repo_000000250.safetensors" in remaining
        assert "repo_000000500.safetensors" not in remaining
        assert not os.path.isdir(samples_dir)
