/**
 * Unzip embeddings from committed .zip file
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import { existsSync } from 'fs';

const execAsync = promisify(exec);

const EMBEDDINGS_DIR = 'public/embeddings';
const ZIP_FILE = `${EMBEDDINGS_DIR}/embeddings.zip`;
const JSON_FILE = `${EMBEDDINGS_DIR}/embeddings.json`;

async function unzipEmbeddings() {
    console.log('📦 Unzipping embeddings...');

    // Check if zip file exists
    if (!existsSync(ZIP_FILE)) {
        console.error(`❌ Error: ${ZIP_FILE} not found!`);
        console.error('   Please ensure embeddings.zip is committed to git.');
        process.exit(1);
    }

    try {
        // Unzip the embeddings
        await execAsync(`unzip -o ${ZIP_FILE} -d ${EMBEDDINGS_DIR}`);

        // Verify extraction
        if (!existsSync(JSON_FILE)) {
            throw new Error('embeddings.json not found after extraction');
        }

        console.log(`✅ Successfully unzipped embeddings`);
        console.log(`   File: ${JSON_FILE}`);
    } catch (error) {
        console.error('❌ Failed to unzip embeddings:', error);
        process.exit(1);
    }
}

unzipEmbeddings();
