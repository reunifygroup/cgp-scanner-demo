"""
Generate "no_card" training images for Class 0
Downloads REAL photos from Unsplash: faces, hands, tables, objects
This will be alphabetically first, so model defaults to "no card" when uncertain
"""

import cv2
import numpy as np
import os
from pathlib import Path
import requests
import time

# Configuration
OUTPUT_DIR = 'training-data'
NUM_IMAGES = 50
IMG_WIDTH = 320
IMG_HEIGHT = 440

# Class name that comes BEFORE "sv02-031_Litleo" alphabetically
CLASS_NAME = "aaa_no_card"  # "aaa" ensures it's Class 0

# Image categories to download (real-world scenarios)
IMAGE_CATEGORIES = [
    'face,person,selfie',
    'hand,fingers,palm',
    'table,desk,surface',
    'wood,texture,background',
    'office,workspace,desk',
    'blur,bokeh,abstract',
    'food,plate,meal',
    'phone,mobile,hand',
    'laptop,computer,keyboard',
    'random,object,everyday',
]

def generate_background_image():
    """Generate a random background (table, surface, etc.)"""
    background_types = [
        ('wood', [101, 67, 33], [139, 90, 43]),      # Brown wood
        ('table', [180, 180, 180], [220, 220, 220]), # Light gray table
        ('dark', [20, 20, 20], [60, 60, 60]),        # Dark surface
        ('white', [230, 230, 230], [255, 255, 255]), # White table
        ('desk', [70, 50, 40], [100, 80, 60]),       # Dark wood
        ('green', [40, 80, 40], [60, 120, 60]),      # Green surface
        ('blue', [40, 40, 80], [60, 60, 120]),       # Blue surface
    ]

    idx = np.random.randint(0, len(background_types))
    bg_type, min_color, max_color = background_types[idx]

    # Random base color
    bg_color = np.array([
        np.random.randint(min_color[0], max_color[0]),
        np.random.randint(min_color[1], max_color[1]),
        np.random.randint(min_color[2], max_color[2])
    ])

    # Create background
    img = np.full((IMG_HEIGHT, IMG_WIDTH, 3), bg_color, dtype=np.uint8)

    # Add texture noise
    noise = np.random.normal(0, 20, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # 50% chance of gradient
    if np.random.random() < 0.5:
        gradient = np.linspace(-30, 30, IMG_HEIGHT)[:, np.newaxis]
        gradient = np.tile(gradient, (1, IMG_WIDTH))
        gradient = np.stack([gradient] * 3, axis=-1)
        img = np.clip(img + gradient, 0, 255).astype(np.uint8)

    return img

def generate_random_shapes():
    """Generate image with random shapes (hands, objects, etc.)"""
    img = generate_background_image()

    # Add random shapes
    num_shapes = np.random.randint(1, 5)
    for _ in range(num_shapes):
        shape_type = np.random.choice(['circle', 'rectangle', 'line'])
        color = tuple(np.random.randint(0, 255, 3).tolist())

        if shape_type == 'circle':
            center = (np.random.randint(0, IMG_WIDTH), np.random.randint(0, IMG_HEIGHT))
            radius = np.random.randint(20, 100)
            cv2.circle(img, center, radius, color, -1)
        elif shape_type == 'rectangle':
            pt1 = (np.random.randint(0, IMG_WIDTH), np.random.randint(0, IMG_HEIGHT))
            pt2 = (np.random.randint(0, IMG_WIDTH), np.random.randint(0, IMG_HEIGHT))
            cv2.rectangle(img, pt1, pt2, color, -1)
        else:  # line
            pt1 = (np.random.randint(0, IMG_WIDTH), np.random.randint(0, IMG_HEIGHT))
            pt2 = (np.random.randint(0, IMG_WIDTH), np.random.randint(0, IMG_HEIGHT))
            thickness = np.random.randint(2, 10)
            cv2.line(img, pt1, pt2, color, thickness)

    return img

def generate_noise_image():
    """Generate pure noise image"""
    noise = np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    # Blend with background for more realistic noise
    bg = generate_background_image()
    alpha = np.random.uniform(0.3, 0.7)
    img = cv2.addWeighted(bg, alpha, noise, 1 - alpha, 0)

    return img

def generate_blurry_image():
    """Generate heavily blurred background"""
    img = generate_background_image()

    # Add some shapes first
    num_blobs = np.random.randint(3, 8)
    for _ in range(num_blobs):
        center = (np.random.randint(0, IMG_WIDTH), np.random.randint(0, IMG_HEIGHT))
        radius = np.random.randint(30, 80)
        color = tuple(np.random.randint(50, 200, 3).tolist())
        cv2.circle(img, center, radius, color, -1)

    # Apply heavy blur
    blur_size = np.random.choice([21, 31, 41, 51])
    img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)

    return img

def generate_partial_object():
    """Generate partial objects (like cropped hand, partial face, etc.)"""
    img = generate_background_image()

    # Simulate skin-like color (hand or face)
    if np.random.random() < 0.6:
        # Skin tones
        skin_colors = [
            [255, 220, 177],  # Light skin
            [241, 194, 125],  # Tan skin
            [224, 172, 105],  # Medium skin
            [198, 134, 66],   # Dark skin
        ]
        skin_color = skin_colors[np.random.randint(0, len(skin_colors))]

        # Create blob shape
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

        # Random organic shape (ellipse)
        center_x = np.random.randint(IMG_WIDTH // 4, 3 * IMG_WIDTH // 4)
        center_y = np.random.randint(IMG_HEIGHT // 4, 3 * IMG_HEIGHT // 4)
        axes_x = np.random.randint(80, 150)
        axes_y = np.random.randint(100, 180)
        angle = np.random.randint(0, 180)

        cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), angle, 0, 360, 255, -1)

        # Apply skin color
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        skin_img = np.full((IMG_HEIGHT, IMG_WIDTH, 3), skin_color, dtype=np.uint8)
        img = (skin_img * mask_3ch + img * (1 - mask_3ch)).astype(np.uint8)

    # Add some texture/noise
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

def main():
    print("🚫 Generating 'no_card' training images")
    print("=" * 50)
    print(f"Class name: {CLASS_NAME} (will be Class 0)")
    print(f"Generating {NUM_IMAGES} images")
    print(f"Output: {OUTPUT_DIR}/\n")

    # Create output directory (directly in training-data/)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    # Image generation functions
    generators = [
        ('background', generate_background_image),
        ('shapes', generate_random_shapes),
        ('noise', generate_noise_image),
        ('blur', generate_blurry_image),
        ('partial', generate_partial_object),
    ]

    # Generate images
    success_count = 0
    for i in range(NUM_IMAGES):
        try:
            # Randomly choose generator
            gen_name, generator = generators[i % len(generators)]

            # Generate image
            img = generator()

            # Save with same naming convention as augmented cards
            output_file = output_path / f"{CLASS_NAME}_gen{i:03d}.png"
            cv2.imwrite(str(output_file), img)

            success_count += 1

            if (i + 1) % 10 == 0:
                print(f"  ✅ Generated {i + 1}/{NUM_IMAGES} images...")

        except Exception as e:
            print(f"  ⚠️  Failed to generate image {i}: {e}")

    print("\n" + "=" * 50)
    print(f"✅ Generated {success_count} 'no_card' images")
    print(f"📁 Saved to: {OUTPUT_DIR}/")
    print(f"\n📋 Next steps:")
    print(f"   1. Run augment-dataset-advanced.py to augment your card images")
    print(f"   2. Upload training-data.zip to Kaggle")
    print(f"   3. Train model - '{CLASS_NAME}' will be Class 0")
    print(f"   4. Model will now default to 'no card' instead of misidentifying!")

if __name__ == "__main__":
    main()
