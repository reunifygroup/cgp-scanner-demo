/**
 * Fetch embeddings from production deployment
 * Used when SKIP_PIPELINE=true to avoid regenerating embeddings
 */

import { mkdir, writeFile } from "fs/promises";
import { existsSync } from "fs";

const PRODUCTION_URL = process.env.PRODUCTION_URL || "https://cgp-scanner-demo.vercel.app";
const EMBEDDINGS_DIR = "public/embeddings";

async function fetchProductionEmbeddings() {
    console.log("📥 Fetching embeddings from production...");
    console.log(`   Production URL: ${PRODUCTION_URL}`);

    try {
        // Create embeddings directory if it doesn't exist
        if (!existsSync(EMBEDDINGS_DIR)) {
            await mkdir(EMBEDDINGS_DIR, { recursive: true });
        }

        // Fetch embeddings.json from production
        console.log("   Downloading embeddings.json...");
        const response = await fetch(`${PRODUCTION_URL}/embeddings/embeddings.json`);

        if (!response.ok) {
            throw new Error(`Failed to fetch embeddings: ${response.status} ${response.statusText}`);
        }

        const embeddings = await response.json();

        // Save to local file
        await writeFile(`${EMBEDDINGS_DIR}/embeddings.json`, JSON.stringify(embeddings, null, 2));

        console.log(`✅ Successfully fetched embeddings from production`);
        console.log(`   Total cards: ${embeddings.total_cards}`);
        console.log(`   Total images: ${embeddings.total_images}`);
        console.log(`   File saved to: ${EMBEDDINGS_DIR}/embeddings.json`);
    } catch (error) {
        console.error("❌ Failed to fetch embeddings from production:", error);
        console.error("\n⚠️  Make sure:");
        console.error("   1. You have a working production deployment");
        console.error("   2. PRODUCTION_URL is set correctly");
        console.error("   3. The production deployment has embeddings");
        process.exit(1);
    }
}

fetchProductionEmbeddings();
