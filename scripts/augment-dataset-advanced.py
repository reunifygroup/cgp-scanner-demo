"""
Simplified, realistic card dataset augmentation for webcam-style recognition.

Key ideas vs previous version:
- Gentler geometric + color augmentations (better for tiny datasets).
- Two background modes:
    * place_card_in_scene: card smaller in frame, lots of visible table/background.
    * add_background: card big, small margin of background around edges.
- No reliance on black background pixels.
"""

import albumentations as A
import cv2
import numpy as np
import os
from pathlib import Path
import random
import shutil

# Configuration
IMAGES_DIR = "images"        # input: one or more base images per card
OUTPUT_DIR = "training-data" # output for training script
AUGMENTATIONS_PER_IMAGE = 25  # how many augmented variants per input image

# Card dimensions (target size for training)
TARGET_WIDTH = 320
TARGET_HEIGHT = 440


# -----------------------------------------------------------------------------
# 1. Augmentation pipeline (simplified & more realistic)
# -----------------------------------------------------------------------------
def create_augmentation_pipeline():
    """
    Create a realistic augmentation pipeline that stays reasonably close to
    real phone camera conditions, without going overboard.

    This is intentionally simpler than the previous "advanced" pipeline:
    - Small rotations / translations / scale changes.
    - Occasional mild blur.
    - Mild brightness/contrast changes.
    - Light noise.
    """

    return A.Compose(
        [
            # Light geometric jitter (no crazy perspective)
            A.Affine(
                rotate=(-10, 10),
                shear=(-5, 5),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                p=0.7,
            ),

            # Mild blur sometimes (out-of-focus / tiny motion)
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ],
                p=0.3,
            ),

            # Subtle brightness / contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5,
            ),

            # Light sensor noise
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        ]
    )


# -----------------------------------------------------------------------------
# 2. Background helpers
# -----------------------------------------------------------------------------
def _generate_table_like_background(height, width):
    """
    Generate a simple "table / desk / surface" style background with
    mild texture and optional gradient.
    """
    background_types = [
        ("wood", [101, 67, 33], [139, 90, 43]),      # Brown wood
        ("table", [180, 180, 180], [220, 220, 220]), # Light gray table
        ("dark", [20, 20, 20], [60, 60, 60]),        # Dark surface
        ("white", [230, 230, 230], [255, 255, 255]), # White table
        ("desk", [70, 50, 40], [100, 80, 60]),       # Dark wood
    ]

    bg_type, min_color, max_color = random.choice(background_types)

    # Random base color within range
    bg_color = np.array(
        [
            np.random.randint(min_color[0], max_color[0]),
            np.random.randint(min_color[1], max_color[1]),
            np.random.randint(min_color[2], max_color[2]),
        ]
    )

    background = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Add subtle texture noise
    noise = np.random.normal(0, 15, background.shape)
    background = np.clip(background + noise, 0, 255).astype(np.uint8)

    # 30% chance of a vertical gradient to simulate lighting
    if random.random() < 0.3:
        gradient = np.linspace(-20, 20, height)[:, np.newaxis]
        gradient = np.tile(gradient, (1, width))
        gradient = np.stack([gradient] * 3, axis=-1)
        background = np.clip(background + gradient, 0, 255).astype(np.uint8)

    return background


def _load_or_generate_background(height, width, backgrounds_dir="backgrounds"):
    """
    Either load a random background image from a folder, or generate a
    synthetic table-like surface if none exist.
    """
    if os.path.exists(backgrounds_dir):
        bg_files = list(Path(backgrounds_dir).glob("*.jpg")) + list(
            Path(backgrounds_dir).glob("*.png")
        )
        if bg_files:
            bg_path = random.choice(bg_files)
            background = cv2.imread(str(bg_path))
            if background is not None:
                background = cv2.resize(background, (width, height))
                return background

    # Fallback: synthetic background
    return _generate_table_like_background(height, width)


def add_background(card_img, backgrounds_dir="backgrounds"):
    """
    Place the card on a background where the card is large and fills most
    of the frame, with a small visible margin around it.

    This is meant to simulate the "card fills the scan window" scenario.
    """
    card_h, card_w = card_img.shape[:2]

    # Create background canvas same size as target/card
    canvas = _load_or_generate_background(card_h, card_w, backgrounds_dir)

    # Card should take ~85–95% of the frame width
    scale_factor = random.uniform(0.85, 0.95)
    new_w = int(card_w * scale_factor)
    new_h = int(card_h * scale_factor)

    resized_card = cv2.resize(card_img, (new_w, new_h))

    # Center with small random jitter
    max_offset_x = card_w - new_w
    max_offset_y = card_h - new_h

    offset_x = max_offset_x // 2
    offset_y = max_offset_y // 2

    jitter_x = int(max_offset_x * 0.2) if max_offset_x > 0 else 0
    jitter_y = int(max_offset_y * 0.2) if max_offset_y > 0 else 0

    offset_x += random.randint(-jitter_x, jitter_x) if jitter_x > 0 else 0
    offset_y += random.randint(-jitter_y, jitter_y) if jitter_y > 0 else 0

    offset_x = max(0, min(offset_x, card_w - new_w))
    offset_y = max(0, min(offset_y, card_h - new_h))

    y1, y2 = offset_y, offset_y + new_h
    x1, x2 = offset_x, offset_x + new_w

    # Paste card on background (no mask needed, full rectangle)
    canvas[y1:y2, x1:x2] = resized_card

    return canvas


def place_card_in_scene(card_img, backgrounds_dir="backgrounds"):
    """
    Place card smaller on a background, so the frame includes a lot of the
    surroundings. This simulates a real camera view where the card is in the
    middle but not filling the entire image.
    """
    card_h, card_w = card_img.shape[:2]

    # Random scale: card takes 60–90% of the frame width (smaller than add_background)
    scale_factor = random.uniform(0.6, 0.9)
    new_card_w = int(card_w * scale_factor)
    new_card_h = int(card_h * scale_factor)

    small_card = cv2.resize(card_img, (new_card_w, new_card_h))

    # Canvas same size as target/card
    canvas_w = card_w
    canvas_h = card_h

    canvas = _load_or_generate_background(canvas_h, canvas_w, backgrounds_dir)

    # Center position with small random offset
    max_offset_x = canvas_w - new_card_w
    max_offset_y = canvas_h - new_card_h

    base_x = max_offset_x // 2
    base_y = max_offset_y // 2

    jitter_x = int(max_offset_x * 0.4) if max_offset_x > 0 else 0
    jitter_y = int(max_offset_y * 0.4) if max_offset_y > 0 else 0

    offset_x = base_x + (random.randint(-jitter_x, jitter_x) if jitter_x > 0 else 0)
    offset_y = base_y + (random.randint(-jitter_y, jitter_y) if jitter_y > 0 else 0)

    offset_x = max(0, min(offset_x, canvas_w - new_card_w))
    offset_y = max(0, min(offset_y, canvas_h - new_card_h))

    y1, y2 = offset_y, offset_y + new_card_h
    x1, x2 = offset_x, offset_x + new_card_w

    canvas[y1:y2, x1:x2] = small_card

    return canvas


# -----------------------------------------------------------------------------
# 3. Per-card processing
# -----------------------------------------------------------------------------
def process_card_image(image_path, output_dir, card_id, augmentation_pipeline):
    """Process a single card base image with augmentation + backgrounds."""

    # Read base image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"   ⚠️  Could not read {image_path}")
        return 0

    # Resize to target dimensions once
    image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save original (no augmentation)
    original_path = os.path.join(output_dir, f"{card_id}_original.png")
    cv2.imwrite(original_path, image)

    success_count = 1  # counting original

    for i in range(AUGMENTATIONS_PER_IMAGE):
        try:
            # Apply core augmentations
            augmented = augmentation_pipeline(image=image)
            augmented_image = augmented["image"]

            # Choose background strategy:
            #  - ~50%: card smaller in a broader scene
            #  - ~50%: card big with small margin of background
            if random.random() < 0.5:
                augmented_image = place_card_in_scene(augmented_image)
            else:
                augmented_image = add_background(augmented_image)

            # Save augmented image
            output_path = os.path.join(output_dir, f"{card_id}_aug{i}.png")
            cv2.imwrite(output_path, augmented_image)
            success_count += 1

        except Exception as e:
            print(f"   ⚠️  Aug {i} failed: {e}")
            continue

    return success_count


# -----------------------------------------------------------------------------
# 4. Main script
# -----------------------------------------------------------------------------
def main():
    print("🎴 Card Dataset Augmentation")
    print("=" * 50)
    print("\n✨ Features:")
    print("   - Gentle affine transforms (rotation, scale, small shifts)")
    print("   - Mild blur and noise (camera-like)")
    print("   - Realistic table/desk backgrounds")
    print("   - Card placed big or small in scene\n")
    print("=" * 50 + "\n")

    # Clean previous training data
    if os.path.exists(OUTPUT_DIR):
        print("🧹 Removing previous training-data directory...")
        shutil.rmtree(OUTPUT_DIR)
        print("✅ Previous data cleaned\n")

    # Create augmentation pipeline
    augmentation_pipeline = create_augmentation_pipeline()

    # Ensure output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images_path = Path(IMAGES_DIR)
    if not images_path.exists():
        print(f"❌ Error: {IMAGES_DIR} directory not found!")
        return

    # Collect all base images in images/
    card_images = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
    if not card_images:
        print("❌ No images found in images/ directory!")
        print("   Please add .png or .jpg files directly to the images/ folder")
        return

    print(f"📦 Processing {len(card_images)} base images (cards)\n")

    total_cards = 0
    total_images = 0

    # Process each base image (each is treated as one card ID)
    for img_path in card_images:
        card_id = img_path.stem  # filename without extension

        print(f"  🎴 Augmenting: {img_path.name}...")

        count = process_card_image(
            img_path,
            OUTPUT_DIR,
            card_id,
            augmentation_pipeline,
        )

        print(f"    ✅ Generated {count} images")
        total_images += count
        total_cards += 1

    print("\n" + "=" * 50)
    print("✅ Augmentation complete!")
    print(f"📊 Total distinct base images (cards) processed: {total_cards}")
    print(f"📸 Total training images (including originals): {total_images}")
    print(f"💾 Output directory: {OUTPUT_DIR}")
    print("\n📋 Next steps:")
    print("   1. Zip the training-data directory")
    print("   2. Upload to Kaggle / Colab")
    print("   3. Train with your train_card_classifier.py")
    print("\n💡 Tip:")
    print("   Add multiple real photos per card into images/ to dramatically")
    print("   improve realism. Each file in images/ becomes its own base.")
    print("=" * 50)


if __name__ == "__main__":
    main()