// scripts/generate_embeddings.js
//
// Generate mean embedding per card from *multiple* augmented images
// using MobileNet (via @tensorflow-models/mobilenet).
//
// Usage:
//   node scripts/generate_embeddings.js
//
// Input:
//   ./training-data/*.png|jpg   (output of your Python augmentation script)
//
// Output:
//   ./card_embeddings.json      (then move to public/model/card_embeddings.json)

import fs from "fs";
import path from "path";
import tf from "@tensorflow/tfjs-node";
import * as mobilenet from "@tensorflow-models/mobilenet";
import { fileURLToPath } from "url";

// ESM __dirname shim
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const PROJECT_ROOT = path.join(__dirname, "..");
const INPUT_DIR = path.join(PROJECT_ROOT, "training-data");
const OUTPUT_PATH = path.join(PROJECT_ROOT, "card_embeddings.json");

// Optional: basic resize before feeding to MobileNet (it will still resize internally)
function preprocessImage(imageTensor) {
    return tf.tidy(() => {
        const resized = tf.image.resizeBilinear(imageTensor, [224, 224]); // [224, 224, 3]
        return resized.toFloat().div(255.0);
    });
}

// Extract cardId from filename stem, e.g.:
//  "sv02-031_Litleo_original" -> "sv02-031_Litleo"
//  "sv02-031_Litleo_aug12"    -> "sv02-031_Litleo"
function getCardIdFromStem(stem) {
    const parts = stem.split("_");
    const last = parts[parts.length - 1];

    if (last === "original" || last.startsWith("aug")) {
        return parts.slice(0, -1).join("_");
    }

    // Fallback: no suffix
    return stem;
}

async function main() {
    console.log("🎴 Generating *mean* embeddings per card from augmented images");
    console.log("=".repeat(60));
    console.log(`📂 Input directory: ${INPUT_DIR}`);
    console.log(`📄 Output JSON:    ${OUTPUT_PATH}\n`);

    if (!fs.existsSync(INPUT_DIR)) {
        console.error(`❌ training-data/ directory not found at:\n${INPUT_DIR}`);
        process.exit(1);
    }

    const allFiles = fs.readdirSync(INPUT_DIR).filter((f) => /\.(png|jpe?g)$/i.test(f));

    if (allFiles.length === 0) {
        console.error("❌ No PNG or JPG files found in training-data/");
        process.exit(1);
    }

    // Group filenames by cardId
    const groups = {};
    for (const file of allFiles) {
        const stem = path.parse(file).name; // remove extension
        const cardId = getCardIdFromStem(stem);
        if (!groups[cardId]) {
            groups[cardId] = [];
        }
        groups[cardId].push(file);
    }

    console.log("📦 Found card groups:");
    for (const [cardId, files] of Object.entries(groups)) {
        console.log(`   • ${cardId}: ${files.length} images`);
    }
    console.log();

    console.log("⬇️  Loading MobileNet feature extractor…\n");
    const model = await mobilenet.load({
        version: 2,
        alpha: 1.0,
    });
    console.log("✅ MobileNet loaded.\n");

    const finalEmbeddings = {}; // { cardId: number[] }

    for (const [cardId, files] of Object.entries(groups)) {
        console.log(`🔍 Processing cardId = "${cardId}" with ${files.length} images`);

        let sumVec = null;
        let count = 0;

        for (const file of files) {
            const filePath = path.join(INPUT_DIR, file);

            try {
                const buffer = fs.readFileSync(filePath);
                const imageTensor = tf.node.decodeImage(buffer, 3);
                const preprocessed = preprocessImage(imageTensor);

                const embeddingTensor = model.infer(preprocessed, { embedding: true });
                const emb1d = embeddingTensor.squeeze();

                const embArray = Array.from(emb1d.dataSync());

                if (!sumVec) {
                    sumVec = embArray.slice(); // clone
                } else {
                    for (let i = 0; i < sumVec.length; i++) {
                        sumVec[i] += embArray[i];
                    }
                }
                count++;

                imageTensor.dispose();
                preprocessed.dispose();
                embeddingTensor.dispose();
                emb1d.dispose();
            } catch (err) {
                console.warn(`   ⚠️  Failed on file ${file}:`, err);
            }
        }

        if (!sumVec || count === 0) {
            console.warn(`   ⚠️  No valid embeddings for cardId "${cardId}", skipping.`);
            continue;
        }

        // Compute mean embedding
        const meanVec = sumVec.map((v) => v / count);
        finalEmbeddings[cardId] = meanVec;

        console.log(`   ✅ Mean embedding computed (length = ${meanVec.length})\n`);
    }

    if (Object.keys(finalEmbeddings).length === 0) {
        console.error("❌ No embeddings computed at all – something went wrong.");
        process.exit(1);
    }

    fs.writeFileSync(OUTPUT_PATH, JSON.stringify(finalEmbeddings, null, 2));
    console.log("💾 Mean embeddings saved to:");
    console.log("   " + OUTPUT_PATH);

    console.log("\nNext steps:");
    console.log("  → mv card_embeddings.json public/model/card_embeddings.json");
    console.log("  → restart dev server if needed, test scanner again.\n");
}

main().catch((err) => {
    console.error("❌ Error:", err);
    process.exit(1);
});
