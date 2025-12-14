/**
 * Simplified, realistic card dataset augmentation for webcam-style recognition.
 * TypeScript version using Sharp for image processing.
 *
 * Key features:
 * - Gentle geometric transformations (rotation, scale, translation)
 * - Realistic color augmentations (brightness, contrast)
 * - Blur and noise effects
 * - Two background placement modes:
 *   * place_card_in_scene: card smaller in frame, lots of visible background
 *   * add_background: card large, small margin of background around edges
 */

import sharp from 'sharp';
import { readdir, mkdir, rm } from 'fs/promises';
import { existsSync } from 'fs';
import { join } from 'path';

// Configuration
const IMAGES_DIR = 'images'; // input: base card images
const OUTPUT_DIR = 'training-data'; // output for training
const AUGMENTATIONS_PER_IMAGE = 10; // variants per input image (optimized from 25)

// Card dimensions (target size for training)
const TARGET_WIDTH = 320;
const TARGET_HEIGHT = 440;

// Background types with color ranges
const BACKGROUND_TYPES = [
  { name: 'wood', minColor: [101, 67, 33], maxColor: [139, 90, 43] },
  { name: 'table', minColor: [180, 180, 180], maxColor: [220, 220, 220] },
  { name: 'dark', minColor: [20, 20, 20], maxColor: [60, 60, 60] },
  { name: 'white', minColor: [230, 230, 230], maxColor: [255, 255, 255] },
  { name: 'desk', minColor: [70, 50, 40], maxColor: [100, 80, 60] },
];

/**
 * Generate a random integer between min and max (inclusive)
 */
function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Generate a random float between min and max
 */
function randomFloat(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

/**
 * Generate a table-like background with texture and optional gradient
 */
async function generateTableBackground(
  width: number,
  height: number
): Promise<Buffer> {
  const bgType =
    BACKGROUND_TYPES[Math.floor(Math.random() * BACKGROUND_TYPES.length)];

  // Random base color within range
  const r = randomInt(bgType.minColor[0], bgType.maxColor[0]);
  const g = randomInt(bgType.minColor[1], bgType.maxColor[1]);
  const b = randomInt(bgType.minColor[2], bgType.maxColor[2]);

  // Create base background
  const background = sharp({
    create: {
      width,
      height,
      channels: 3,
      background: { r, g, b },
    },
  });

  // Add subtle texture noise (simulate using slight blur and modulate)
  let pipeline = background
    .blur(0.3) // Very subtle blur for texture
    .modulate({
      brightness: randomFloat(0.95, 1.05),
      saturation: randomFloat(0.9, 1.1),
    });

  // 30% chance of adding a gradient overlay
  if (Math.random() < 0.3) {
    // Create a gradient overlay using linear gradient
    const gradient = Buffer.from(
      Array(width * height * 4)
        .fill(0)
        .map((_, i) => {
          const pixelIndex = Math.floor(i / 4);
          const y = Math.floor(pixelIndex / width);
          const alpha = i % 4;

          if (alpha === 3) return 255; // Alpha channel

          // Vertical gradient from darker at top to lighter at bottom
          const gradientValue = ((y / height) * 40 - 20) * 0.3;
          return Math.max(0, Math.min(255, gradientValue));
        })
    );

    const gradientImage = sharp(gradient, {
      raw: { width, height, channels: 4 },
    });

    pipeline = pipeline.composite([
      {
        input: await gradientImage.png().toBuffer(),
        blend: 'add',
      },
    ]);
  }

  return pipeline.png().toBuffer();
}

/**
 * Apply augmentations to a card image - OPTIMIZED VERSION
 * Focus on essential transformations that match real-world mobile scanning
 */
async function applyAugmentations(cardBuffer: Buffer): Promise<Buffer> {
  let augmented = sharp(cardBuffer);
  const metadata = await augmented.metadata();

  // ALWAYS apply rotation (essential for mobile scanning)
  // Rotation (-10° to +10°)
  const rotation = randomFloat(-10, 10);
  augmented = augmented.rotate(rotation, {
    background: { r: 128, g: 128, b: 128, alpha: 0 },
  });

  // 50% chance of blur (reduced from 30% but kept for realism)
  if (Math.random() < 0.5) {
    const blurType = Math.random();
    if (blurType < 0.5) {
      // Gaussian blur (mild)
      augmented = augmented.blur(randomFloat(0.3, 1.0));
    } else {
      // Motion blur simulation (mild)
      augmented = augmented.blur(randomFloat(0.5, 1.5));
    }
  }

  // ALWAYS apply brightness/saturation adjustment (essential for lighting variations)
  const brightness = randomFloat(0.85, 1.15); // Slightly wider range for better coverage

  augmented = augmented.modulate({
    brightness,
    saturation: randomFloat(0.9, 1.1), // Moderate saturation variation
  });

  // Removed: Noise augmentation (minimal real-world value)

  // Ensure we maintain size after transformations and release resources
  return augmented
    .resize(metadata.width!, metadata.height!, {
      fit: 'cover',
      position: 'center',
    })
    .removeAlpha() // Ensure RGB (3 channels) output
    .toBuffer();
}

/**
 * Place card on background where card fills most of the frame (85-95%)
 */
async function addBackground(cardBuffer: Buffer): Promise<Buffer> {
  const cardImage = sharp(cardBuffer);

  // Generate background
  const background = await generateTableBackground(
    TARGET_WIDTH,
    TARGET_HEIGHT
  );

  // Card should take 85-95% of frame
  const scaleFactor = randomFloat(0.85, 0.95);
  const newWidth = Math.round(TARGET_WIDTH * scaleFactor);
  const newHeight = Math.round(TARGET_HEIGHT * scaleFactor);

  const resizedCard = await cardImage
    .resize(newWidth, newHeight, { fit: 'fill' })
    .toBuffer();

  // Center with small random jitter
  const maxOffsetX = TARGET_WIDTH - newWidth;
  const maxOffsetY = TARGET_HEIGHT - newHeight;

  const baseX = Math.floor(maxOffsetX / 2);
  const baseY = Math.floor(maxOffsetY / 2);

  const jitterX = Math.floor(maxOffsetX * 0.2);
  const jitterY = Math.floor(maxOffsetY * 0.2);

  const offsetX = Math.max(
    0,
    Math.min(
      baseX + randomInt(-jitterX, jitterX),
      TARGET_WIDTH - newWidth
    )
  );
  const offsetY = Math.max(
    0,
    Math.min(
      baseY + randomInt(-jitterY, jitterY),
      TARGET_HEIGHT - newHeight
    )
  );

  // Composite card onto background
  return sharp(background)
    .composite([
      {
        input: resizedCard,
        top: offsetY,
        left: offsetX,
      },
    ])
    .removeAlpha() // Ensure RGB (3 channels) output
    .png()
    .toBuffer();
}

/**
 * Place card smaller in scene (60-90% of frame width)
 */
async function placeCardInScene(cardBuffer: Buffer): Promise<Buffer> {
  const cardImage = sharp(cardBuffer);

  // Generate background
  const background = await generateTableBackground(
    TARGET_WIDTH,
    TARGET_HEIGHT
  );

  // Card takes 60-90% of frame
  const scaleFactor = randomFloat(0.6, 0.9);
  const newWidth = Math.round(TARGET_WIDTH * scaleFactor);
  const newHeight = Math.round(TARGET_HEIGHT * scaleFactor);

  const resizedCard = await cardImage
    .resize(newWidth, newHeight, { fit: 'fill' })
    .toBuffer();

  // Center with random offset
  const maxOffsetX = TARGET_WIDTH - newWidth;
  const maxOffsetY = TARGET_HEIGHT - newHeight;

  const baseX = Math.floor(maxOffsetX / 2);
  const baseY = Math.floor(maxOffsetY / 2);

  const jitterX = Math.floor(maxOffsetX * 0.4);
  const jitterY = Math.floor(maxOffsetY * 0.4);

  const offsetX = Math.max(
    0,
    Math.min(
      baseX + randomInt(-jitterX, jitterX),
      TARGET_WIDTH - newWidth
    )
  );
  const offsetY = Math.max(
    0,
    Math.min(
      baseY + randomInt(-jitterY, jitterY),
      TARGET_HEIGHT - newHeight
    )
  );

  // Composite card onto background
  return sharp(background)
    .composite([
      {
        input: resizedCard,
        top: offsetY,
        left: offsetX,
      },
    ])
    .removeAlpha() // Ensure RGB (3 channels) output
    .png()
    .toBuffer();
}

/**
 * Process a single card image with augmentation
 */
async function processCardImage(
  imagePath: string,
  outputDir: string,
  cardId: string
): Promise<number> {
  try {
    // Read and resize base image
    const image = await sharp(imagePath)
      .resize(TARGET_WIDTH, TARGET_HEIGHT, { fit: 'fill' })
      .toBuffer();

    // Save original (no augmentation)
    const originalPath = join(outputDir, `${cardId}_original.png`);
    await sharp(image).removeAlpha().png().toFile(originalPath);

    let successCount = 1; // counting original

    // Generate augmented versions
    for (let i = 0; i < AUGMENTATIONS_PER_IMAGE; i++) {
      try {
        // Apply core augmentations
        let augmented = await applyAugmentations(image);

        // Choose background strategy (90/10 - optimized for mobile app cropping)
        // 90% use addBackground (85-95% fill) - matches mobile app reality
        // 10% use placeCardInScene (60-90% fill) - edge cases only
        if (Math.random() < 0.1) {
          augmented = await placeCardInScene(augmented);
        } else {
          augmented = await addBackground(augmented);
        }

        // Save augmented image
        const outputPath = join(outputDir, `${cardId}_aug${i}.png`);
        await sharp(augmented).removeAlpha().png().toFile(outputPath);
        successCount++;
      } catch (err) {
        console.error(`   ⚠️  Aug ${i} failed:`, err);
        continue;
      }
    }

    return successCount;
  } catch (err) {
    console.error(`   ⚠️  Could not process ${imagePath}:`, err);
    return 0;
  }
}

/**
 * Main augmentation script
 */
async function main() {
  console.log('🎴 Card Dataset Augmentation (TypeScript + Sharp) - OPTIMIZED');
  console.log('='.repeat(50));
  console.log('\n✨ Features (Optimized for Mobile Scanning):');
  console.log('   - Essential transforms: rotation (always), blur (50%)');
  console.log('   - Brightness/saturation variation (always)');
  console.log('   - Realistic backgrounds (90% close-up, 10% distance)');
  console.log('   - 10 augmentations per card (down from 25)');
  console.log('   - Expected: ~3,100 total embeddings');
  console.log('   - Expected search speedup: ~2.5x faster\n');
  console.log('='.repeat(50) + '\n');

  // Clean previous training data
  if (existsSync(OUTPUT_DIR)) {
    console.log('🧹 Removing previous training-data directory...');
    await rm(OUTPUT_DIR, { recursive: true });
    console.log('✅ Previous data cleaned\n');
  }

  // Create output directory
  await mkdir(OUTPUT_DIR, { recursive: true });

  // Check if images directory exists
  if (!existsSync(IMAGES_DIR)) {
    console.error(`❌ Error: ${IMAGES_DIR} directory not found!`);
    return;
  }

  // Collect all base images
  const files = await readdir(IMAGES_DIR);
  const cardImages = files.filter(
    (file) => file.endsWith('.png') || file.endsWith('.jpg')
  );

  if (cardImages.length === 0) {
    console.error('❌ No images found in images/ directory!');
    console.error(
      '   Please add .png or .jpg files to the images/ folder'
    );
    return;
  }

  console.log(`📦 Processing ${cardImages.length} base images (cards)\n`);

  let totalCards = 0;
  let totalImages = 0;

  // Process each base image
  for (const imageFile of cardImages) {
    const cardId = imageFile.replace(/\.(png|jpg)$/i, '');
    const imagePath = join(IMAGES_DIR, imageFile);

    console.log(`  🎴 Augmenting: ${imageFile}...`);

    const count = await processCardImage(imagePath, OUTPUT_DIR, cardId);

    console.log(`    ✅ Generated ${count} images`);
    totalImages += count;
    totalCards++;

    // Force garbage collection every 10 cards to manage memory
    if (totalCards % 10 === 0 && global.gc) {
      global.gc();
    }
  }

  console.log('\n' + '='.repeat(50));
  console.log('✅ Augmentation complete!');
  console.log(`📊 Total base images (cards) processed: ${totalCards}`);
  console.log(
    `📸 Total training images (including originals): ${totalImages}`
  );
  console.log(`💾 Output directory: ${OUTPUT_DIR}`);
  console.log('\n📋 Next steps:');
  console.log('   1. Run: npm run generate-embeddings');
  console.log('   2. Start your scanner app');
  console.log('\n💡 Tip:');
  console.log(
    '   Add multiple real photos per card to images/ to improve accuracy.'
  );
  console.log('='.repeat(50));
}

// Run the script
main().catch((error) => {
  console.error('❌ Fatal error:', error);
  process.exit(1);
});
