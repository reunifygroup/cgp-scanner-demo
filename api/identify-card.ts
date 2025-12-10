import type { VercelRequest, VercelResponse } from '@vercel/node';
import * as tf from '@tensorflow/tfjs-node';
import * as mobilenet from '@tensorflow-models/mobilenet';
import { readFileSync } from 'fs';
import { join } from 'path';

interface EmbeddingData {
    card_id: string;
    filename: string;
    embedding: number[];
}

interface EmbeddingsDatabase {
    model: string;
    embedding_dim: number;
    total_images: number;
    total_cards: number;
    embeddings: EmbeddingData[];
}

// Cache model and embeddings in memory
let cachedModel: mobilenet.MobileNet | null = null;
let cachedEmbeddings: EmbeddingsDatabase | null = null;

/**
 * Load MobileNet model (cached)
 */
async function loadModel(): Promise<mobilenet.MobileNet> {
    if (!cachedModel) {
        console.log('Loading MobileNet model...');
        cachedModel = await mobilenet.load({
            version: 2,
            alpha: 1.0,
        });
        console.log('MobileNet model loaded');
    }
    return cachedModel;
}

/**
 * Load embeddings database (cached)
 */
function loadEmbeddings(): EmbeddingsDatabase {
    if (!cachedEmbeddings) {
        console.log('Loading embeddings database...');
        const embeddingsPath = join(process.cwd(), 'public', 'embeddings', 'embeddings.json');
        const data = readFileSync(embeddingsPath, 'utf-8');
        cachedEmbeddings = JSON.parse(data);
        console.log(`Embeddings loaded: ${cachedEmbeddings.total_images} images, ${cachedEmbeddings.total_cards} cards`);
    }
    return cachedEmbeddings;
}

/**
 * Compute cosine similarity between two vectors
 */
function cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
        throw new Error('Vectors must have same length');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) {
        return 0;
    }

    return dotProduct / (normA * normB);
}

/**
 * Find nearest card in embeddings database
 */
function findNearestCard(queryEmbedding: number[], embeddings: EmbeddingsDatabase): { cardId: string; similarity: number } {
    let bestMatch = {
        cardId: '',
        similarity: -1,
    };

    for (const entry of embeddings.embeddings) {
        const similarity = cosineSimilarity(queryEmbedding, entry.embedding);

        if (similarity > bestMatch.similarity) {
            bestMatch = {
                cardId: entry.card_id,
                similarity: similarity,
            };
        }
    }

    return bestMatch;
}

/**
 * Extract embedding from image
 */
async function extractEmbedding(model: mobilenet.MobileNet, imageBuffer: Buffer): Promise<number[]> {
    // Decode image from buffer
    const imageTensor = tf.node.decodeImage(imageBuffer, 3) as tf.Tensor3D;

    // Resize to 224x224
    const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);

    // Extract embedding
    const embeddingTensor = model.infer(resized as any, true) as tf.Tensor;

    // Get embedding as array
    const embeddingArray = await embeddingTensor.data();
    const embedding = Array.from(embeddingArray);

    // L2 normalize
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    const normalizedEmbedding = embedding.map((val) => val / norm);

    // Cleanup
    imageTensor.dispose();
    resized.dispose();
    embeddingTensor.dispose();

    return normalizedEmbedding;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
    // Only allow POST
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    try {
        // Get image from request body (base64 encoded)
        const { image } = req.body;

        if (!image) {
            return res.status(400).json({ error: 'No image provided' });
        }

        // Decode base64 image
        const imageBuffer = Buffer.from(image.split(',')[1], 'base64');

        // Load model and embeddings
        const model = await loadModel();
        const embeddings = loadEmbeddings();

        // Extract embedding from uploaded image
        const queryEmbedding = await extractEmbedding(model, imageBuffer);

        // Find nearest match
        const match = findNearestCard(queryEmbedding, embeddings);

        // Parse card name from ID
        const cardName = match.cardId.split('_').slice(1).join(' ');

        // Return result
        return res.status(200).json({
            cardId: match.cardId,
            cardName: cardName,
            similarity: match.similarity * 100, // Convert to percentage
        });
    } catch (error) {
        console.error('Error identifying card:', error);
        return res.status(500).json({
            error: 'Failed to identify card',
            details: error instanceof Error ? error.message : 'Unknown error',
        });
    }
}
