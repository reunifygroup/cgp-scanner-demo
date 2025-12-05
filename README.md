# 🎴 Pokémon Card Scanner (CNN-based)

Real-time Pokémon TCG card recognition using Convolutional Neural Networks (CNN) with TensorFlow.js.

## 🚀 Features

-   **AI-Powered Recognition**: Deep learning CNN model for instant card identification
-   **Real-time Scanning**: Browser-based inference with TensorFlow.js
-   **Anywhere Detection**: Recognizes cards anywhere in frame, any angle/lighting
-   **No Backend Needed**: All inference happens in the browser
-   **Easy Training**: Simple workflow from images to trained model

## 📋 Complete Workflow

### Step 1: Prepare Card Images

Place your card images in the `images/` directory, organized by set:

```
images/
├── sv09/
│   └── sv09-001_Caterpie.png
└── sv10/
    └── sv10-001_Ethan.png
```

**Note**: You can start with just 2 cards for quick testing! More cards = longer training time.

### Step 2: Generate Augmented Dataset

**IMPORTANT**: Use the advanced augmentation for production-quality results:

```bash
# Install Python dependencies (first time only)
pip3 install -r requirements.txt

# Run advanced augmentation (RECOMMENDED)
npm run augment-advanced
```

This creates `training-data/` with 50 high-quality variations per card including:
- **3D perspective transforms** (card at angles)
- **Realistic lighting** (shadows, highlights, glare)
- **Camera simulation** (blur, noise, focus issues)
- **Partial occlusion** (simulates fingers, glare spots)
- **Background integration** (card on different surfaces)

**Alternative** (basic augmentation, not recommended):
```bash
npm run augment-dataset  # Simple augmentation, lower quality
```

### Step 3: Create Training ZIP

Zip the training data for upload to Google Colab:

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cd training-data && zip -r ../training-data_${TIMESTAMP}.zip . && cd ..
rm -rf training-data
```

### Step 4: Train Model in Google Colab

1. **Open Google Colab**: https://colab.research.google.com/
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Working directory and reset**: Navigate or create the working directory under `/content/cgpremium/scanner`
4. **Upload & Extract Training Data**:

    ```python
    from google.colab import files
    import os
    import glob

    # Upload zip file (any name works)
    uploaded = files.upload()

    # Find the uploaded zip file
    zip_files = glob.glob('*.zip')
    if not zip_files:
        raise Exception("No zip file found!")

    zip_file = max(zip_files, key=os.path.getctime)
    print(f"Found: {zip_file}")

    # Force remove old training-data (always)
    print("Cleaning up old training-data...")
    !rm -rf training-data

    # Unzip to training-data directory
    print(f"Unzipping {zip_file}...")
    !unzip -q {zip_file} -d training-data

    # Remove the uploaded zip file
    print(f"Removing {zip_file}...")
    !rm {zip_file}

    print("✅ Ready for training!")
    ```

5. **Copy & Paste** the entire `train_card_classifier.py` script into a cell
6. **Run the cell** - Training will start automatically
7. **Download** the auto-generated `card_classifier_model.zip` when done
8. **Place** the zip contents under `public/model`in the client directory

## 🛠️ Scripts Reference

| Command                    | Description                     |
| -------------------------- | ------------------------------- |
| `npm run augment-dataset`  | Generate 20 variations per card |
| `cd client && npm run dev` | Start scanner app               |

## 📊 Training Details

-   **Model**: Custom CNN (3 conv blocks, optimized for small datasets)
-   **Input Size**: 224x224 RGB
-   **Training Time**: ~2-5 min per card on Colab GPU
-   **Model Size**: ~2MB
-   **Accuracy**: Depends on training data quality (see below)

## 🎯 How It Works

1. **Camera Capture**: Captures frame every 500ms
2. **Preprocessing**: Resize to 224x224, normalize to [0,1]
3. **CNN Inference**: TensorFlow.js runs model in browser
4. **Prediction**: Returns card ID with confidence score
5. **Display**: Shows result when confidence > 70%

## 🔧 Technical Stack

### Frontend

-   **React + TypeScript** - UI framework
-   **Vite** - Build tool
-   **TensorFlow.js** - Browser ML inference
-   **WebRTC getUserMedia** - Camera access

### Training

-   **Python 3** - Training scripts
-   **TensorFlow/Keras 3** - Deep learning framework
-   **MobileNetV2** - Pre-trained CNN base
-   **Sharp** - Image augmentation
-   **Google Colab** - Free GPU training

## 📝 Model Architecture

```
Input (224x224x3)
    ↓
MobileNetV2 Base (frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(128, relu)
    ↓
Dropout(0.2)
    ↓
Dense(num_cards, softmax)
    ↓
Output (card probabilities)
```

## 🎓 How This Works at Scale (4 Cards → 20,000 Cards)

### ✅ Production Apps (TCGPlayer, Delver Lens, etc.)

These apps recognize 20,000+ cards using:

1. **Official card images only** (from APIs like yours)
2. **Advanced augmentation** (perspective, lighting, camera simulation)
3. **CNN classification** (same approach you're using)
4. **Scale advantage** (20k cards = tons of training data)

**Key**: They use sophisticated augmentation to simulate real camera conditions from flat images.

### 🎯 Your Setup

**With Advanced Augmentation** (`augment-advanced`):
- ✅ Simulates 3D perspective (card at angles)
- ✅ Realistic lighting (shadows, glare, highlights)
- ✅ Camera effects (blur, noise, compression)
- ✅ Partial occlusion (fingers, other cards)
- ✅ Works with official card images only

**Results**:
- **4 cards (testing)**: Should work decently with advanced augmentation
- **100 cards**: Good accuracy (more data to learn from)
- **1,000+ cards**: Great accuracy (scale helps generalization)
- **20,000 cards**: Production-quality (like the big apps)

### 📊 Expected Performance

| Cards | Augmentation | Expected Accuracy | Notes |
|-------|-------------|-------------------|-------|
| 4 | Basic | 20-30% | Model overfits, memorizes patterns |
| 4 | Advanced | 60-75% | Better, but limited data |
| 100 | Advanced | 80-90% | Good for demos |
| 1,000+ | Advanced | 90-95% | Production-ready |
| 20,000 | Advanced | 95%+ | Pro-level like big apps |

**Why scale matters**: With more cards, the model learns general card features instead of memorizing specific patterns.

## 🎓 Tips for Best Results

1. **Start Small**: Test with 2 cards first to verify pipeline
2. **Quality > Quantity**: 10 diverse photos > 100 augmentations of 1 photo
3. **Good Images**: Use high-quality card images (400x560+ px)
4. **Real Photos**: Take actual photos with your phone camera
5. **More Data**: 20+ real photos per card for production accuracy

## 📄 License

ISC
