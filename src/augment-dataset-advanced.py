"""
Advanced Card Dataset Augmentation for Production-Scale Recognition
Uses sophisticated transforms to simulate real camera conditions
"""

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import numpy as np
import os
from pathlib import Path
import random
import shutil

# Configuration
IMAGES_DIR = 'images'
OUTPUT_DIR = 'training-data'
AUGMENTATIONS_PER_IMAGE = 50  # Generate 50 high-quality variations per card

# Card dimensions (target size for training)
TARGET_WIDTH = 400
TARGET_HEIGHT = 560

# Custom edge-only dropout transform
class EdgeCoarseDropout(ImageOnlyTransform):
    """CoarseDropout that only places holes on the edges (not center)"""

    def __init__(
        self,
        max_holes=3,
        max_height=50,
        max_width=50,
        min_holes=1,
        min_height=20,
        min_width=20,
        fill_value=0,
        edge_margin=0.25,  # Only place holes in outer 25% of image
        always_apply=False,
        p=0.5
    ):
        super().__init__(always_apply, p)
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes
        self.min_height = min_height
        self.min_width = min_width
        self.fill_value = fill_value
        self.edge_margin = edge_margin

    def apply(self, img, **params):
        height, width = img.shape[:2]
        holes = random.randint(self.min_holes, self.max_holes)

        for _ in range(holes):
            hole_height = random.randint(self.min_height, self.max_height)
            hole_width = random.randint(self.min_width, self.max_width)

            # Randomly choose which edge: 0=top, 1=right, 2=bottom, 3=left
            edge = random.randint(0, 3)

            if edge == 0:  # Top edge
                y1 = random.randint(0, int(height * self.edge_margin))
                x1 = random.randint(0, max(1, width - hole_width))
            elif edge == 1:  # Right edge
                y1 = random.randint(0, max(1, height - hole_height))
                x1 = random.randint(int(width * (1 - self.edge_margin)), max(1, width - hole_width))
            elif edge == 2:  # Bottom edge
                y1 = random.randint(int(height * (1 - self.edge_margin)), max(1, height - hole_height))
                x1 = random.randint(0, max(1, width - hole_width))
            else:  # Left edge
                y1 = random.randint(0, max(1, height - hole_height))
                x1 = random.randint(0, int(width * self.edge_margin))

            y2 = min(y1 + hole_height, height)
            x2 = min(x1 + hole_width, width)

            img[y1:y2, x1:x2] = self.fill_value

        return img

def create_advanced_augmentation_pipeline():
    """
    Create production-quality augmentation pipeline that simulates
    real-world camera conditions from flat card images
    """

    return A.Compose([
        # 1. PERSPECTIVE & GEOMETRY - Simulate card at different angles
        A.OneOf([
            A.Perspective(scale=(0.05, 0.15), p=1.0),  # Card viewed at angle
            A.Affine(
                rotate=(-25, 25),
                shear=(-15, 15),
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                scale=(0.8, 1.2),
                p=1.0
            ),
        ], p=0.9),

        # 2. REALISTIC LIGHTING - Simulate different lighting conditions
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.RandomToneCurve(scale=0.3, p=1.0),  # Natural lighting variation
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=30,
                p=1.0
            ),
        ], p=0.8),

        # 3. SHADOWS & HIGHLIGHTS - Simulate uneven lighting
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.3),
            contrast_limit=0.2,
            p=0.4
        ),

        # 4. CAMERA EFFECTS - Simulate phone camera issues
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),  # Focus issues
            A.MotionBlur(blur_limit=7, p=1.0),  # Camera shake
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),  # Sensor noise
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),

        # 5. EXPOSURE & WHITE BALANCE (removed extreme color transforms)
        A.OneOf([
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),  # Contrast enhancement
        ], p=0.3),

        # 7. COMPRESSION ARTIFACTS - Simulate compressed images
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),

        # 8. EDGE DROPOUT - Only on borders, preserves center features
        EdgeCoarseDropout(
            max_holes=3,
            max_height=50,
            max_width=50,
            min_holes=1,
            min_height=20,
            min_width=20,
            fill_value=0,
            edge_margin=0.25,  # Only outer 25% of image (edges)
            p=0.2
        ),

    ])

def add_background(card_img, backgrounds_dir='backgrounds'):
    """
    Place card on random background to simulate real environment
    """
    # If no backgrounds provided, create colored noise background
    if not os.path.exists(backgrounds_dir):
        bg_color = np.random.randint(50, 200, 3)
        noise = np.random.normal(0, 30, card_img.shape)
        background = np.full(card_img.shape, bg_color, dtype=np.uint8)
        background = np.clip(background + noise, 0, 255).astype(np.uint8)
    else:
        # Use random background image
        bg_files = list(Path(backgrounds_dir).glob('*.jpg')) + \
                   list(Path(backgrounds_dir).glob('*.png'))
        if bg_files:
            bg_path = random.choice(bg_files)
            background = cv2.imread(str(bg_path))
            background = cv2.resize(background, (card_img.shape[1], card_img.shape[0]))
        else:
            return card_img

    # Create mask for card (assume card is centered)
    mask = np.any(card_img != [0, 0, 0], axis=-1).astype(np.uint8) * 255

    # Blend card onto background
    mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
    result = (card_img * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)

    return result

def process_card_image(image_path, output_dir, card_id, augmentation_pipeline):
    """Process a single card image with advanced augmentation"""

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"   ⚠️  Could not read {image_path}")
        return 0

    # Resize to target dimensions
    image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save original (no augmentation)
    original_path = os.path.join(output_dir, f"{card_id}_original.png")
    cv2.imwrite(original_path, image)

    # Generate augmented versions
    success_count = 1  # Count original

    for i in range(AUGMENTATIONS_PER_IMAGE):
        try:
            # Apply augmentation pipeline
            augmented = augmentation_pipeline(image=image)
            augmented_image = augmented['image']

            # Optionally add background (20% chance)
            if random.random() < 0.2:
                augmented_image = add_background(augmented_image)

            # Save augmented image
            output_path = os.path.join(output_dir, f"{card_id}_aug{i}.png")
            cv2.imwrite(output_path, augmented_image)
            success_count += 1

        except Exception as e:
            print(f"   ⚠️  Aug {i} failed: {e}")
            continue

    return success_count

def main():
    print("🎴 Advanced Card Dataset Augmentation")
    print("=" * 50)
    print("\n✨ Features:")
    print("   - Perspective transforms (3D card angles)")
    print("   - Realistic lighting & shadows")
    print("   - Camera blur & noise")
    print("   - Partial occlusion simulation")
    print("   - Background integration")
    print("\n" + "=" * 50 + "\n")

    # Clean up previous training data if exists
    if os.path.exists(OUTPUT_DIR):
        print("🧹 Removing previous training-data directory...")
        shutil.rmtree(OUTPUT_DIR)
        print("✅ Previous data cleaned\n")

    # Create augmentation pipeline
    augmentation_pipeline = create_advanced_augmentation_pipeline()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process all sets
    images_path = Path(IMAGES_DIR)
    if not images_path.exists():
        print(f"❌ Error: {IMAGES_DIR} directory not found!")
        return

    total_cards = 0
    total_images = 0

    # Get all images directly in images/ directory
    card_images = list(images_path.glob('*.png')) + list(images_path.glob('*.jpg'))

    if not card_images:
        print("❌ No images found in images/ directory!")
        print("   Please add .png or .jpg files directly to the images/ folder")
        return

    print(f"📦 Processing {len(card_images)} cards\n")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each card
    for img_path in card_images:
        # Extract card ID from filename
        card_id = img_path.stem

        print(f"  🎴 Augmenting: {img_path.name}...")

        # Process card
        count = process_card_image(
            img_path,
            OUTPUT_DIR,
            card_id,
            augmentation_pipeline
        )

        print(f"    ✅ Generated {count} images")
        total_images += count
        total_cards += 1

    print("\n" + "=" * 50)
    print("✅ Augmentation complete!")
    print(f"📊 Total cards processed: {total_cards}")
    print(f"📸 Total training images: {total_images}")
    print(f"💾 Output directory: {OUTPUT_DIR}")
    print("\n📋 Next steps:")
    print("   1. Zip the training-data directory")
    print("   2. Upload to Google Colab")
    print("   3. Train with train_card_classifier.py")
    print("\n💡 This augmentation simulates real camera conditions!")
    print("   Works with official card images - no physical photos needed")

if __name__ == "__main__":
    main()
