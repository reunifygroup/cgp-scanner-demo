import sharp from 'sharp';
import fs from 'fs/promises';
import path from 'path';

// 🎨 Configuration
const IMAGES_DIR = path.join(process.cwd(), 'images');
const OUTPUT_DIR = path.join(process.cwd(), 'training-data');
const AUGMENTATIONS_PER_IMAGE = 100; // Generate 100 variations per card

// 🔄 Augmentation functions
const augmentations = [
  // Rotation variations
  async (img: sharp.Sharp) => img.rotate(5),
  async (img: sharp.Sharp) => img.rotate(-5),
  async (img: sharp.Sharp) => img.rotate(10),
  async (img: sharp.Sharp) => img.rotate(-10),

  // Brightness variations
  async (img: sharp.Sharp) => img.modulate({ brightness: 1.2 }),
  async (img: sharp.Sharp) => img.modulate({ brightness: 0.8 }),
  async (img: sharp.Sharp) => img.modulate({ brightness: 1.3 }),
  async (img: sharp.Sharp) => img.modulate({ brightness: 0.7 }),

  // Contrast variations
  async (img: sharp.Sharp) => img.linear(1.2, 0),
  async (img: sharp.Sharp) => img.linear(0.8, 0),

  // Saturation variations
  async (img: sharp.Sharp) => img.modulate({ saturation: 1.3 }),
  async (img: sharp.Sharp) => img.modulate({ saturation: 0.7 }),

  // Blur variations (simulate focus issues)
  async (img: sharp.Sharp) => img.blur(0.5),
  async (img: sharp.Sharp) => img.blur(1),

  // Sharpen
  async (img: sharp.Sharp) => img.sharpen(),

  // Scale variations
  async (img: sharp.Sharp) => img.resize(450, 630).resize(400, 560),
  async (img: sharp.Sharp) => img.resize(380, 530).resize(400, 560),

  // Noise (simulate camera noise)
  async (img: sharp.Sharp) => img.median(3),

  // Combined augmentations
  async (img: sharp.Sharp) => img.rotate(7).modulate({ brightness: 1.1 }),
  async (img: sharp.Sharp) => img.rotate(-7).modulate({ brightness: 0.9 }),
];

// 🎯 Apply random augmentation
const applyRandomAugmentation = async (imagePath: string, outputPath: string, augIndex: number) => {
  try {
    let img = sharp(imagePath);

    // Apply 1-3 random augmentations
    const numAugmentations = Math.min(1 + Math.floor(augIndex / 7), 3);
    const selectedAugmentations = [];

    for (let i = 0; i < numAugmentations; i++) {
      const randomAug = augmentations[Math.floor(Math.random() * augmentations.length)];
      selectedAugmentations.push(randomAug);
    }

    // Apply augmentations sequentially
    for (const aug of selectedAugmentations) {
      img = await aug(img);
    }

    // Ensure consistent output size
    await img
      .resize(400, 560, { fit: 'fill' })
      .toFile(outputPath);

    return true;
  } catch (error) {
    console.error(`Failed to augment ${imagePath}:`, error);
    return false;
  }
};

// 📁 Process all cards
const processCards = async () => {
  console.log('🎨 Card Dataset Augmentation\n');
  console.log('='.repeat(50));

  try {
    // Create output directory structure
    await fs.mkdir(OUTPUT_DIR, { recursive: true });

    // Read all set directories
    const sets = await fs.readdir(IMAGES_DIR, { withFileTypes: true });
    const setDirs = sets.filter(entry => entry.isDirectory()).map(entry => entry.name);

    let totalCards = 0;
    let totalImages = 0;

    for (const setId of setDirs) {
      console.log(`\n📦 Processing set: ${setId}`);

      const setInputDir = path.join(IMAGES_DIR, setId);
      const setOutputDir = path.join(OUTPUT_DIR, setId);

      await fs.mkdir(setOutputDir, { recursive: true });

      // Get all card images
      const files = await fs.readdir(setInputDir);
      const imageFiles = files.filter(f => f.endsWith('.png') || f.endsWith('.jpg'));

      for (const fileName of imageFiles) {
        const inputPath = path.join(setInputDir, fileName);
        const baseName = fileName.replace(/\.(png|jpg)$/, '');

        console.log(`  🎴 Augmenting: ${fileName}...`);

        // Copy original
        const originalOutput = path.join(setOutputDir, `${baseName}_original.png`);
        await sharp(inputPath)
          .resize(400, 560, { fit: 'fill' })
          .toFile(originalOutput);

        // Generate augmentations
        let successCount = 1; // Count original
        for (let i = 0; i < AUGMENTATIONS_PER_IMAGE; i++) {
          const augmentedOutput = path.join(setOutputDir, `${baseName}_aug${i}.png`);
          const success = await applyRandomAugmentation(inputPath, augmentedOutput, i);
          if (success) successCount++;
        }

        console.log(`    ✅ Generated ${successCount} images`);
        totalImages += successCount;
        totalCards++;
      }
    }

    console.log('\n' + '='.repeat(50));
    console.log('✅ Augmentation complete!');
    console.log(`📊 Total cards processed: ${totalCards}`);
    console.log(`📸 Total training images: ${totalImages}`);
    console.log(`💾 Output directory: ${OUTPUT_DIR}`);
    console.log('\n📋 Next step: Create training notebook for Google Colab');

  } catch (error) {
    console.error('💥 Fatal error:', error);
    process.exit(1);
  }
};

// 🎯 Execute
processCards();
