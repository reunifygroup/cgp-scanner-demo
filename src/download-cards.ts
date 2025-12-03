import axios from "axios";
import fs from "fs/promises";
import path from "path";

// 📦 Types for TCGdex API responses
interface Card {
    id: string;
    localId: string;
    name: string;
    image?: string;
}

interface SetResponse {
    id: string;
    name: string;
    cards: Card[];
}

// 🌐 TCGdex API base URL
const TCGDEX_API = "https://api.tcgdex.net/v2/en";
const IMAGES_DIR = path.join(process.cwd(), "images");
const LIMIT_PER_SET = 1;

// 🔽 Download a single image from URL
const downloadImage = async (imageUrl: string, filePath: string): Promise<void> => {
    try {
        const response = await axios.get(imageUrl, {
            responseType: "arraybuffer",
            timeout: 10000,
        });

        await fs.writeFile(filePath, response.data);
        console.log(`✅ Downloaded: ${path.basename(filePath)}`);
    } catch (error) {
        console.error(`❌ Failed to download ${imageUrl}:`, error instanceof Error ? error.message : error);
    }
};

// 📥 Fetch set data and download card images
const downloadSetImages = async (setId: string): Promise<void> => {
    try {
        console.log(`\n🔍 Fetching set: ${setId}...`);

        // Fetch set data from TCGdex API
        const { data: setData } = await axios.get<SetResponse>(`${TCGDEX_API}/sets/${setId}`);

        console.log(`📦 Set: ${setData.name} (${setData.cards.length} cards total)`);

        // Create directory for this set
        const setDir = path.join(IMAGES_DIR, setId);
        await fs.mkdir(setDir, { recursive: true });

        // Limit to first N cards
        const cardsToDownload = setData.cards.slice(0, LIMIT_PER_SET);
        console.log(`⬇️  Downloading ${cardsToDownload.length} cards...\n`);

        // Download images sequentially to avoid overwhelming the server
        for (const card of cardsToDownload) {
            // Construct image URL based on TCGdex structure
            const imageUrl = `${card.image}/low.png`;
            const fileName = `${card.id}_${card.name.replace(/[^a-z0-9]/gi, "_")}.png`;
            const filePath = path.join(setDir, imageUrl);

            await downloadImage(fileName, filePath);

            // Small delay to be respectful to the API
            await new Promise((resolve) => setTimeout(resolve, 100));
        }

        console.log(`\n✨ Completed downloading ${setId}\n`);
    } catch (error) {
        console.error(`❌ Error processing set ${setId}:`, error instanceof Error ? error.message : error);
    }
};

// 🚀 Main execution function
const main = async () => {
    console.log("🎴 TCGdex Card Image Downloader\n");
    console.log("=".repeat(50));

    // Target sets: sv09 = Journey Together, sv10 = Destined Rivals
    const sets = ["sv09", "sv10"];

    // Create base images directory
    await fs.mkdir(IMAGES_DIR, { recursive: true });

    // Download images for each set
    for (const setId of sets) {
        await downloadSetImages(setId);
    }

    console.log("✅ All downloads complete!");
    console.log(`📁 Images saved to: ${IMAGES_DIR}`);
};

// 🎯 Execute script
main().catch((error) => {
    console.error("💥 Fatal error:", error);
    process.exit(1);
});
