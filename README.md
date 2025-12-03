# 🎴 Pokémon Card Scanner

A real-time Pokémon TCG card scanner using perceptual hashing and computer vision.

## 🚀 Features

- **Fast Recognition**: Uses perceptual hashing for instant card identification
- **Real-time Scanning**: Continuous video capture with automatic matching
- **Web-based**: React frontend with camera access
- **REST API**: Fastify backend with hash matching
- **434 Cards**: Pre-loaded with sv09 (Journey Together) and sv10 (Destined Rivals) sets

## 📦 Project Structure

```
tcgdex/
├── api-server.js           # Fastify API with card matching endpoint
├── src/
│   ├── download-cards.ts   # Script to download card images from TCGdex
│   └── generate-hashes.ts  # Script to generate perceptual hashes
├── client/                 # React frontend (Vite + TypeScript)
│   └── src/
│       ├── App.tsx         # Main scanner component
│       └── App.css         # Styles
├── images/                 # Downloaded card images
│   ├── sv09/              # Journey Together set
│   └── sv10/              # Destined Rivals set
└── hash-database.json      # Pre-computed perceptual hashes for all cards
```

## 🔧 Setup

### 1. Download Card Images

```bash
npm run download
```

This downloads images for sv09 and sv10 sets from TCGdex API.

### 2. Generate Hash Database

```bash
npm run generate-hashes
```

This creates `hash-database.json` with perceptual hashes for all downloaded cards.

### 3. Install Frontend Dependencies

```bash
cd client
npm install
cd ..
```

## 🎮 Running the Scanner

You need to run TWO servers:

### Terminal 1: Start the API Server

```bash
npm run api
```

The API will start on `http://localhost:3000`

### Terminal 2: Start the Frontend

```bash
cd client
npm run dev
```

The frontend will start on `http://localhost:5173` (or similar)

## 📱 Using the Scanner

1. Open the frontend URL in your browser
2. Click **"📸 Start Scanner"** button
3. Allow camera access when prompted
4. Point your camera at a Pokémon card
5. The scanner will automatically identify the card and display:
   - Card ID (e.g., "sv09-001")
   - Card Name
   - Set ID
   - Confidence score

## 🔍 How It Works

1. **Video Capture**: Frontend captures video frames every 500ms
2. **Frame Extraction**: Converts video frame to PNG image
3. **API Request**: Sends image to backend `/api/scan` endpoint
4. **Hash Generation**: Backend generates perceptual hash of the image
5. **Matching**: Compares hash against 434 pre-computed hashes using Hamming distance
6. **Result**: Returns best match with confidence score

## 🛠️ Technical Stack

### Backend
- **Fastify** - Web framework
- **imghash** - Perceptual hashing library
- **sharp** - Image processing
- **axios** - HTTP client for TCGdex API

### Frontend
- **React** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **WebRTC getUserMedia** - Camera access

### Matching Algorithm
- **Perceptual Hashing (pHash)** - 16-bit hash for image similarity
- **Hamming Distance** - Bitwise comparison for fast matching
- **Threshold**: Max distance of 20 for matches

## 📊 API Endpoints

### `GET /api/health`
Check API status and card database

### `POST /api/scan`
Upload image and get card match

**Request**: Multipart form data with `file` field

**Response**:
```json
{
  "matched": true,
  "card": {
    "cardId": "sv09-001",
    "cardName": "Caterpie",
    "setId": "sv09",
    "distance": 3,
    "confidence": 85.0
  },
  "alternatives": [...],
  "totalMatches": 5
}
```

## 🎯 Scripts Reference

| Command | Description |
|---------|-------------|
| `npm run download` | Download card images from TCGdex |
| `npm run generate-hashes` | Generate perceptual hash database |
| `npm run api` | Start Fastify API server |
| `cd client && npm run dev` | Start React dev server |

## 🔒 Security Notes

- CORS is set to allow all origins (adjust for production)
- Camera access requires HTTPS in production
- No authentication implemented (add for production use)

## 📝 License

ISC
