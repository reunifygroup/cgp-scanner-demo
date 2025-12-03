import fs from 'fs/promises';
import path from 'path';
import imghash from 'imghash';

// 📦 Types for hash database
interface CardHash {
  cardId: string;
  cardName: string;
  setId: string;
  fileName: string;
  hashes: {
    perceptual: string;  // pHash - best for variations in lighting/compression
    difference: string;  // dHash - good for detecting slight changes
    average: string;     // aHash - fastest, good for duplicates
  };
}

interface HashDatabase {
  generatedAt: string;
  totalCards: number;
  sets: string[];
  cards: CardHash[];
}

// 🌐 Configuration
const IMAGES_DIR = path.join(process.cwd(), 'images');
const OUTPUT_FILE = path.join(process.cwd(), 'hash-database.json');

// 🔢 Generate all hash types for an image
const generateHashes = async (imagePath: string): Promise<CardHash['hashes']> => {
  const [perceptual, difference, average] = await Promise.all([
    imghash.hash(imagePath, 16, 'hex'),  // 16-bit pHash
    imghash.hash(imagePath, 16, 'hex'),  // Using same for now
    imghash.hash(imagePath, 8, 'hex'),   // 8-bit aHash (faster)
  ]);

  return {
    perceptual,
    difference,
    average,
  };
};

// 📁 Process all images in a set directory
const processSet = async (setId: string): Promise<CardHash[]> => {
  const setDir = path.join(IMAGES_DIR, setId);
  const cardHashes: CardHash[] = [];

  try {
    const files = await fs.readdir(setDir);
    const imageFiles = files.filter(f => f.endsWith('.png') || f.endsWith('.jpg'));

    console.log(`\n📦 Processing set: ${setId} (${imageFiles.length} images)`);

    for (const fileName of imageFiles) {
      try {
        const imagePath = path.join(setDir, fileName);

        // Extract card info from filename: {id}_{name}.png
        const [cardId, ...nameParts] = fileName.replace(/\.(png|jpg)$/, '').split('_');
        const cardName = nameParts.join(' ').replace(/_/g, ' ');

        console.log(`  🔢 Hashing: ${fileName}...`);

        // Generate hashes
        const hashes = await generateHashes(imagePath);

        cardHashes.push({
          cardId,
          cardName,
          setId,
          fileName,
          hashes,
        });

      } catch (error) {
        console.error(`  ❌ Failed to process ${fileName}:`, error instanceof Error ? error.message : error);
      }
    }

    console.log(`  ✅ Completed ${setId}: ${cardHashes.length} cards processed`);

  } catch (error) {
    console.error(`❌ Error reading set directory ${setId}:`, error instanceof Error ? error.message : error);
  }

  return cardHashes;
};

// 🚀 Main execution function
const main = async () => {
  console.log('🔢 Card Hash Generator\n');
  console.log('='.repeat(50));

  try {
    // Read all set directories
    const entries = await fs.readdir(IMAGES_DIR, { withFileTypes: true });
    const setDirs = entries
      .filter(entry => entry.isDirectory())
      .map(entry => entry.name);

    if (setDirs.length === 0) {
      console.error('❌ No set directories found in images/');
      process.exit(1);
    }

    console.log(`\n📂 Found ${setDirs.length} sets: ${setDirs.join(', ')}\n`);

    // Process all sets
    const allCardHashes: CardHash[] = [];
    for (const setId of setDirs) {
      const setHashes = await processSet(setId);
      allCardHashes.push(...setHashes);
    }

    // Create database object
    const database: HashDatabase = {
      generatedAt: new Date().toISOString(),
      totalCards: allCardHashes.length,
      sets: setDirs,
      cards: allCardHashes,
    };

    // Save to JSON file
    await fs.writeFile(OUTPUT_FILE, JSON.stringify(database, null, 2));

    console.log('\n' + '='.repeat(50));
    console.log('✅ Hash generation complete!');
    console.log(`📊 Total cards processed: ${database.totalCards}`);
    console.log(`💾 Database saved to: ${OUTPUT_FILE}`);
    console.log(`📦 Sets included: ${database.sets.join(', ')}`);

  } catch (error) {
    console.error('💥 Fatal error:', error);
    process.exit(1);
  }
};

// 🎯 Execute script
main().catch((error) => {
  console.error('💥 Fatal error:', error);
  process.exit(1);
});
