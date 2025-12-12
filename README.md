# CGP Pokemon Card Scanner

AI-powered instant Pokemon card recognition using computer vision and embedding-based similarity search. Built with TypeScript, TensorFlow.js, and React.

![Made with TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![Deployed on Vercel](https://img.shields.io/badge/Vercel-000000?style=flat&logo=vercel&logoColor=white)

## Features

-   **Real-time card recognition** via webcam
-   **Dual validation system** - combines MobileNet v2 embeddings + Tesseract.js OCR
-   **Fuzzy text matching** - handles OCR errors and misspellings with Fuse.js
-   **Scalable architecture** - handles 20,000+ cards without retraining
-   **100% TypeScript**
-   **Compressed embeddings** - zipped embeddings committed to git for fast deployments
-   **Realistic augmentation** - 26 variations per card with backgrounds, rotation, blur, etc.

## How It Works

This scanner uses a **two-stage validation system** combining embedding-based recognition with OCR verification:

### Stage 1: Embedding-Based Recognition

1. **MobileNet v2** extracts 1024-dimensional feature vectors from card images
2. Pre-computed embeddings for all training images are stored in a database
3. When scanning, the camera frame is:
    - Captured and preprocessed (224x224)
    - Converted to an embedding vector
    - Compared against all database embeddings using **cosine similarity**
4. The **top 3 matches** are retrieved and checked

### Stage 2: OCR Validation

If **all top 3 matches exceed 65% similarity**, the system performs OCR validation:

1. **Image preprocessing**:
    - Crops to top half of card (where Pokemon names appear)
    - Converts to grayscale for better text recognition
    - Applies brightness (+50) and contrast (+60) enhancement

2. **Tesseract.js OCR**:
    - Optimized for single-line text detection
    - Extracts Pokemon name from card image

3. **Fuzzy matching with Fuse.js**:
    - Compares OCR text against Pokemon names database
    - Handles misspellings and OCR errors
    - Threshold: 0.4 (0 = exact match, 1 = match anything)

4. **Final validation**:
    - Checks if OCR text matches any of the top 3 embedding results
    - Both exact substring matching and fuzzy matching are attempted
    - Only confirms a hit if **both vision (embeddings) and text (OCR) agree**

**Why this hybrid approach?**

-   **Embeddings alone** can be fooled by similar-looking cards
-   **OCR validation** ensures the detected Pokemon name matches what's visually recognized
-   **Higher accuracy** with minimal false positives
-   **Robust to variations** in lighting, angle, and card condition

## Quick Start

### Local Development

```bash
# Install dependencies
npm install

# If you have embeddings.zip, unzip it:
npm run unzip-embeddings

# OR run full pipeline to generate embeddings:
npm run pipeline

# Start dev server
npm run dev
```

Open http://localhost:5173 and allow camera access to start scanning!

## Available Scripts

| Script                        | Description                                       |
| ----------------------------- | ------------------------------------------------- |
| `npm run dev`                 | Start Vite dev server                             |
| `npm run build`               | **Unzip embeddings + build** (used by Vercel)    |
| `npm run download-cards`      | Download cards from TCGdex API                    |
| `npm run augment`             | Generate augmented training images                |
| `npm run generate-embeddings` | Extract MobileNet embeddings                      |
| `npm run pipeline`            | Download + Augment + Generate + Zip embeddings    |
| `npm run unzip-embeddings`    | Extract embeddings.json from embeddings.zip       |
| `npm run preview`             | Preview production build                          |

## Data Pipeline

### 1. Download Cards

```bash
npm run download-cards
```

Downloads Pokemon card images from the [TCGdex API](https://www.tcgdex.dev/):

-   Fetches high-quality PNG images
-   Configurable limit via `CARD_LIMIT` environment variable
-   Sets to download can be varied within the array in the script
-   Default: 50 cards (configurable in `vercel.json`)
-   **Generates `pokemon-names.json`** - database of all unique Pokemon names for OCR fuzzy matching

### 2. Augment Dataset

```bash
npm run augment
```

Generates realistic training variations using Sharp:

-   **25 augmentations per card** = 26 total images (original + 25)
-   Gentle rotations (+/- 10 degrees), scale (0.9-1.1x), translations
-   Blur effects (Gaussian & motion)
-   Brightness & saturation adjustments
-   Realistic table/desk backgrounds (5 types)
-   Two placement modes:
    -   **Scene mode**: Card at 60-90% size with visible background
    -   **Close-up mode**: Card at 85-95% size with small margins

**Output**: `training-data/` directory with augmented images

### 3. Generate Embeddings

```bash
npm run generate-embeddings
```

Extracts embeddings using MobileNet v2:

-   Loads MobileNet directly from `@tensorflow-models/mobilenet`
-   Processes all training images
-   Extracts 1024-dimensional feature vectors
-   L2-normalizes for cosine similarity
-   Saves to `public/embeddings/embeddings.json`

### 4. Full Pipeline with Auto-Zip

```bash
npm run pipeline
```

Runs all steps and automatically creates a compressed zip file:

-   Downloads cards → Augments → Generates embeddings → **Zips embeddings.json**
-   Creates `public/embeddings/embeddings.zip` (< 100MB when compressed)
-   Ready to commit to git and deploy

## Deployment (Vercel)

The project uses a **zip-based deployment** approach for fast builds:

### How It Works

1. **Generate embeddings locally** (one-time or when updating cards):
   ```bash
   npm run pipeline
   # Creates public/embeddings/embeddings.zip (< 100MB)
   ```

2. **Commit and push**:
   ```bash
   git add public/embeddings/embeddings.zip
   git commit -m "Update embeddings"
   git push
   ```

3. **Vercel deployment**:
   - Unzips embeddings.zip → embeddings.json
   - Builds the Vite app
   - **Fast deployment** (~2-3 minutes)

### Benefits

-   **No heavy files in git** - Only the compressed zip (< 100MB) is committed
-   **Fast Vercel builds** - No need to regenerate embeddings on every deploy
-   **Simple updates** - Just re-run `npm run pipeline` and commit the new zip

**Important**: The zip file is committed to git, but the JSON files are `.gitignore`d. Embeddings are unzipped during the Vercel build process.

## Technology Stack

### Frontend

-   **React 19** - UI framework
-   **TypeScript** - Type safety
-   **Vite** - Build tool & dev server
-   **@tensorflow/tfjs** - MobileNet inference in browser
-   **Tesseract.js** - Client-side OCR for text recognition
-   **Fuse.js** - Fuzzy string matching for OCR validation

### Backend/Scripts

-   **Node.js** - Script runtime
-   **@tensorflow/tfjs-node** - MobileNet for embeddings generation
-   **@tensorflow-models/mobilenet** - Pre-trained MobileNet v2
-   **Sharp** - Image processing (resize, augmentation, backgrounds)
-   **Axios** - HTTP requests to TCGdex API

### Deployment

-   **Vercel** - Hosting & CI/CD
-   **GitHub** - Version control & triggers

---

**Built with love using TensorFlow.js and React**

_Card data provided by [TCGdex](https://www.tcgdex.dev/)_
