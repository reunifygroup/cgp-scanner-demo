import Fastify from 'fastify';
import cors from '@fastify/cors';
import multipart from '@fastify/multipart';
import fs from 'fs/promises';
import path from 'path';
import imghash from 'imghash';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { extractCard } from './card-detector.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// 🚀 Initialize Fastify server
const fastify = Fastify({
  logger: true,
  bodyLimit: 10485760 // 10MB limit for image uploads
});

// 🌐 Enable CORS
await fastify.register(cors, {
  origin: '*' // Allow all origins (adjust for production)
});

// 📦 Enable multipart for file uploads
await fastify.register(multipart);

// 💾 Load hash database
let hashDatabase = null;
const loadDatabase = async () => {
  const dbPath = path.join(__dirname, 'hash-database.json');
  const data = await fs.readFile(dbPath, 'utf-8');
  hashDatabase = JSON.parse(data);
  console.log(`📊 Loaded ${hashDatabase.totalCards} cards from database`);
};

// 🔢 Calculate Hamming distance between two hex strings
const hammingDistance = (hash1, hash2) => {
  if (hash1.length !== hash2.length) {
    throw new Error('Hash lengths must match');
  }

  let distance = 0;
  for (let i = 0; i < hash1.length; i++) {
    const xor = parseInt(hash1[i], 16) ^ parseInt(hash2[i], 16);
    // Count set bits in XOR result
    distance += xor.toString(2).split('1').length - 1;
  }
  return distance;
};

// 🔍 Find best matching cards
const findMatches = async (imageHash, topN = 5, maxDistance = 50) => {
  const matches = [];
  let closestDistance = Infinity;

  for (const card of hashDatabase.cards) {
    // Compare using perceptual hash (best for variations)
    const distance = hammingDistance(imageHash, card.hashes.perceptual);

    // Track closest distance for debugging
    if (distance < closestDistance) {
      closestDistance = distance;
    }

    if (distance <= maxDistance) {
      matches.push({
        cardId: card.cardId,
        cardName: card.cardName,
        setId: card.setId,
        fileName: card.fileName,
        distance,
        confidence: Math.max(0, 100 - (distance * 2)) // Simple confidence score
      });
    }
  }

  // Sort by distance (lower is better)
  matches.sort((a, b) => a.distance - b.distance);

  console.log(`🔍 Closest match distance: ${closestDistance}`);

  return matches.slice(0, topN);
};

// 🌐 Health check endpoint
fastify.get('/api/health', async (request, reply) => {
  return {
    status: 'ok',
    cardsLoaded: hashDatabase?.totalCards || 0,
    timestamp: new Date().toISOString()
  };
});

// 📸 Card scanning endpoint
fastify.post('/api/scan', async (request, reply) => {
  try {
    const data = await request.file();

    if (!data) {
      return reply.code(400).send({ error: 'No image provided' });
    }

    // Save temporary file
    const tempPath = path.join(__dirname, 'temp-scan.png');
    const processedPath = path.join(__dirname, 'temp-processed.png');
    await fs.writeFile(tempPath, await data.toBuffer());

    // Extract and preprocess card
    const extracted = await extractCard(tempPath, processedPath);

    // Generate hash for processed image
    const hashPath = extracted ? processedPath : tempPath;
    const imageHash = await imghash.hash(hashPath, 16, 'hex');
    console.log(`📸 Scan request - Hash: ${imageHash} (extracted: ${extracted})`);

    // Find matches
    const matches = await findMatches(imageHash);

    // Clean up temp files
    await fs.unlink(tempPath).catch(() => {});
    await fs.unlink(processedPath).catch(() => {});

    if (matches.length === 0) {
      console.log(`❌ No match found`);
      return {
        matched: false,
        message: 'No matching card found',
        hash: imageHash
      };
    }

    console.log(`✅ Match found: ${matches[0].cardId} (distance: ${matches[0].distance})`);

    // Return best match
    return {
      matched: true,
      card: matches[0],
      alternatives: matches.slice(1),
      totalMatches: matches.length
    };

  } catch (error) {
    fastify.log.error(error);
    return reply.code(500).send({
      error: 'Failed to process image',
      details: error.message
    });
  }
});

// 🔌 Start the server
const startServer = async () => {
  try {
    // Load hash database
    await loadDatabase();

    // Start server
    await fastify.listen({ port: 3000, host: '0.0.0.0' });
    console.log('🎴 Card Scanner API Ready!');
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

startServer();
