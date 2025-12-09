/**
 * Generate embeddings for card images using TensorFlow.js and MobileNet.
 *
 * This script:
 * 1. Loads MobileNet directly from TensorFlow.js
 * 2. Processes all images in training-data/
 * 3. Extracts fixed-size embeddings (1024-dim vectors from MobileNet)
 * 4. Saves embeddings database as JSON
 * 5. Deploys to client/public/embeddings/
 *
 * The embeddings can then be used for similarity-based card recognition.
 */

import * as tf from '@tensorflow/tfjs-node';
import * as mobilenet from '@tensorflow-models/mobilenet';
import sharp from 'sharp';
import { readdir, mkdir, writeFile, copyFile, stat } from 'fs/promises';
import { existsSync } from 'fs';
import { join } from 'path';

// Configuration
const TRAINING_DATA_DIR = 'training-data'; // augmented images directory
const EMBEDDINGS_OUTPUT_DIR = 'embeddings'; // where to save the results
const CLIENT_EMBEDDINGS_DIR = 'client/public/embeddings'; // where to deploy for web
const IMAGE_SIZE = 224; // MobileNet input size
const EMBEDDING_DIM = 1024; // MobileNet final embedding dimension

interface EmbeddingData {
  card_id: string;
  filename: string;
  embedding: number[];
}

interface EmbeddingsDatabase {
  model: string;
  embedding_dim: number;
  image_size: [number, number];
  total_images: number;
  total_cards: number;
  generated_at: string;
  embeddings: EmbeddingData[];
}

/**
 * Load the pre-trained MobileNet model from TensorFlow.js.
 */
async function loadFeatureExtractor(): Promise<mobilenet.MobileNet> {
  console.log('🤖 Loading MobileNet from TensorFlow.js...');

  // Load MobileNet v2 with version 1.0 (1024-dim embeddings)
  const model = await mobilenet.load({
    version: 2,
    alpha: 1.0, // Full model (not quantized)
  });

  console.log('✅ MobileNet loaded successfully');
  console.log(`   Model: MobileNet v2 (alpha=1.0)`);
  console.log(`   Embedding dimension: 1024`);

  return model;
}

/**
 * Load and preprocess an image for MobileNet.
 *
 * @param imagePath - Path to the image file
 * @returns Preprocessed image tensor ready for model input
 */
async function loadAndPreprocessImage(imagePath: string): Promise<tf.Tensor3D> {
  // Load image with sharp and resize to 224x224
  const imageBuffer = await sharp(imagePath)
    .resize(IMAGE_SIZE, IMAGE_SIZE, { fit: 'fill' })
    .raw() // Get raw pixel data
    .toBuffer({ resolveWithObject: true });

  // Convert buffer to tensor [224, 224, 3]
  const imageTensor = tf.tensor3d(
    new Uint8Array(imageBuffer.data),
    [IMAGE_SIZE, IMAGE_SIZE, 3],
    'int32'
  );

  // Convert to float32 - MobileNet expects pixel values in [0, 255]
  const preprocessed = imageTensor.toFloat();

  // Clean up intermediate tensor
  imageTensor.dispose();

  return preprocessed;
}

/**
 * Extract embedding vector for a single image.
 *
 * @param model - MobileNet model
 * @param imagePath - Path to image
 * @returns 1D array of shape (1024,)
 */
async function extractEmbedding(
  model: mobilenet.MobileNet,
  imagePath: string
): Promise<number[]> {
  const img = await loadAndPreprocessImage(imagePath);

  // Extract embeddings (activations from the layer before final classification)
  const embeddingTensor = model.infer(img, true) as tf.Tensor;

  // Convert to array: [1024]
  const embeddingArray = await embeddingTensor.data();
  const embedding = Array.from(embeddingArray);

  // L2 normalize the embedding for cosine similarity
  const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  const normalizedEmbedding = embedding.map((val) => val / norm);

  // Clean up tensors
  img.dispose();
  embeddingTensor.dispose();

  return normalizedEmbedding;
}

/**
 * Extract card ID from augmented filename.
 *
 * Examples:
 *   sv02-031_Litleo_original.png -> sv02-031_Litleo
 *   sv02-031_Litleo_aug0.png -> sv02-031_Litleo
 *   sv02-185_Iono_aug42.png -> sv02-185_Iono
 */
function parseCardIdFromFilename(filename: string): string {
  // Remove extension
  const stem = filename.replace(/\.(png|jpg|jpeg)$/i, '');

  // Remove suffixes like _original, _aug0, etc.
  if (stem.includes('_original')) {
    return stem.replace('_original', '');
  } else if (stem.includes('_aug')) {
    // Split on _aug and take everything before it
    return stem.split('_aug')[0];
  } else {
    return stem;
  }
}

/**
 * Main function to generate embeddings for all training images.
 */
async function generateEmbeddingsDatabase(): Promise<void> {
  console.log('🎴 Card Embeddings Generator (TypeScript + TensorFlow.js)');
  console.log('='.repeat(60));

  // Check training data exists
  if (!existsSync(TRAINING_DATA_DIR)) {
    console.error(`❌ Error: ${TRAINING_DATA_DIR} directory not found!`);
    console.error('   Please run augmentation first (npm run augment-advanced)');
    return;
  }

  // Collect all images
  const allFiles = await readdir(TRAINING_DATA_DIR);
  const imageFiles = allFiles.filter((file) =>
    /\.(png|jpg|jpeg)$/i.test(file)
  );

  if (imageFiles.length === 0) {
    console.error(`❌ No images found in ${TRAINING_DATA_DIR}`);
    return;
  }

  console.log(`\n📦 Found ${imageFiles.length} training images`);

  // Create output directory
  if (!existsSync(EMBEDDINGS_OUTPUT_DIR)) {
    await mkdir(EMBEDDINGS_OUTPUT_DIR, { recursive: true });
  }

  // Load feature extractor
  const model = await loadFeatureExtractor();
  console.log();

  // Process all images
  const embeddingsData: EmbeddingData[] = [];
  const cardIdCounts: Record<string, number> = {};

  console.log('🔄 Extracting embeddings...');
  const startTime = Date.now();

  for (let i = 0; i < imageFiles.length; i++) {
    const filename = imageFiles[i];
    const imagePath = join(TRAINING_DATA_DIR, filename);

    try {
      // Extract embedding
      const embedding = await extractEmbedding(model, imagePath);

      // Parse card ID from filename
      const cardId = parseCardIdFromFilename(filename);

      // Track card counts
      cardIdCounts[cardId] = (cardIdCounts[cardId] || 0) + 1;

      // Store embedding data
      embeddingsData.push({
        card_id: cardId,
        filename: filename,
        embedding: embedding,
      });

      // Progress indicator
      if ((i + 1) % 10 === 0 || i === imageFiles.length - 1) {
        process.stdout.write(
          `\r   Processed ${i + 1}/${imageFiles.length} images...`
        );
      }
    } catch (error) {
      console.error(`\n⚠️  Failed to process ${filename}:`, error);
      continue;
    }
  }

  const elapsedTime = (Date.now() - startTime) / 1000;
  console.log(
    `\n\n✅ Extracted ${embeddingsData.length} embeddings in ${elapsedTime.toFixed(1)}s`
  );
  console.log(
    `   (${((elapsedTime / embeddingsData.length) * 1000).toFixed(1)}ms per image)`
  );

  // Save embeddings as JSON
  const embeddingsDatabase: EmbeddingsDatabase = {
    model: 'MobileNet_v2_1.0',
    embedding_dim: EMBEDDING_DIM,
    image_size: [IMAGE_SIZE, IMAGE_SIZE],
    total_images: embeddingsData.length,
    total_cards: Object.keys(cardIdCounts).length,
    generated_at: new Date().toISOString().replace('T', ' ').substring(0, 19),
    embeddings: embeddingsData,
  };

  const embeddingsJsonPath = join(EMBEDDINGS_OUTPUT_DIR, 'embeddings.json');
  await writeFile(
    embeddingsJsonPath,
    JSON.stringify(embeddingsDatabase, null, 2)
  );

  const jsonStats = await stat(embeddingsJsonPath);
  console.log(`\n💾 Saved embeddings as JSON: ${embeddingsJsonPath}`);
  console.log(`   Size: ${(jsonStats.size / 1024 / 1024).toFixed(2)} MB`);

  // Save metadata (without embeddings for smaller file)
  const metadata = {
    model: 'MobileNet_v2_1.0',
    embedding_dim: EMBEDDING_DIM,
    image_size: [IMAGE_SIZE, IMAGE_SIZE],
    total_images: embeddingsData.length,
    total_cards: Object.keys(cardIdCounts).length,
    generated_at: embeddingsDatabase.generated_at,
  };

  const metadataPath = join(EMBEDDINGS_OUTPUT_DIR, 'metadata.json');
  await writeFile(metadataPath, JSON.stringify(metadata, null, 2));
  console.log(`💾 Saved metadata: ${metadataPath}`);

  // Copy embeddings.json to client/public/embeddings
  console.log(`\n📦 Deploying to ${CLIENT_EMBEDDINGS_DIR}...`);
  if (!existsSync(CLIENT_EMBEDDINGS_DIR)) {
    await mkdir(CLIENT_EMBEDDINGS_DIR, { recursive: true });
  }

  const clientEmbeddingsPath = join(CLIENT_EMBEDDINGS_DIR, 'embeddings.json');
  await copyFile(embeddingsJsonPath, clientEmbeddingsPath);
  console.log(`✅ Embeddings deployed to ${clientEmbeddingsPath}`);

  // Print summary
  console.log('\n' + '='.repeat(60));
  console.log('📊 Summary:');
  console.log(`   Total unique cards: ${Object.keys(cardIdCounts).length}`);
  console.log(`   Total embeddings: ${embeddingsData.length}`);
  console.log(`   Embedding dimension: ${EMBEDDING_DIM}`);
  console.log();
  console.log('📋 Cards processed:');

  const sortedCards = Object.entries(cardIdCounts).sort((a, b) =>
    a[0].localeCompare(b[0])
  );
  for (const [cardId, count] of sortedCards) {
    console.log(`   • ${cardId}: ${count} variations`);
  }

  console.log('\n✅ Embedding database created and deployed!');
  console.log(`\n📁 Source directory: ${EMBEDDINGS_OUTPUT_DIR}/`);
  console.log('   ├── embeddings.json (JSON format - also deployed to client)');
  console.log('   └── metadata.json (database info)');
  console.log(`\n📁 Deployed to: ${CLIENT_EMBEDDINGS_DIR}/`);
  console.log('   └── embeddings.json (ready for web scanner)');

  console.log('\n🚀 Ready to use! Restart your dev server.');
  console.log('='.repeat(60));
}

// Run the script
generateEmbeddingsDatabase().catch((error) => {
  console.error('❌ Fatal error:', error);
  process.exit(1);
});
