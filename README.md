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
│   ├── sv09-001_Caterpie.png
│   └── sv09-002_Metapod.png
└── sv10/
    ├── sv10-001_Ethan.png
    └── sv10-002_Yanma.png
```

**Note**: You can start with just 2 cards for quick testing! More cards = longer training time.

### Step 2: Generate Augmented Dataset

Run the augmentation script to create 20 variations of each card:

```bash
npm run augment-dataset
```

This creates `training-data/` with augmented images (rotation, brightness, blur, etc.).

### Step 3: Create Training ZIP

Zip the training data for upload to Google Colab:

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
zip -r training-data_${TIMESTAMP}.zip training-data
rm -rf training-data
```

This creates `training-data_20251204_103045.zip` and removes the source directory.

### Step 4: Train Model in Google Colab

1. **Open Google Colab**: https://colab.research.google.com/
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Upload & Extract Training Data**:
    ```python
    from google.colab import files
    import os
    import glob
    import shutil

    # Upload zip file (any name works)
    uploaded = files.upload()

    # Find the uploaded zip file
    zip_files = glob.glob('*.zip')
    if not zip_files:
        raise Exception("No zip file found!")

    zip_file = max(zip_files, key=os.path.getctime)
    print(f"Found: {zip_file}")

    # Remove old training-data if exists
    if os.path.exists('training-data'):
        print("Removing old training-data...")
        shutil.rmtree('training-data')

    # Unzip to training-data directory
    print(f"Unzipping {zip_file}...")
    !unzip -q {zip_file} -d training-data
    print("✅ Ready for training!")
    ```
4. **Copy & Paste** the entire `train_card_classifier.py` script into a cell
6. **Run the cell** - Training will start automatically
7. **Download** the auto-generated `card_classifier_model.zip` when done

## 🛠️ Scripts Reference

| Command                    | Description                     |
| -------------------------- | ------------------------------- |
| `npm run augment-dataset`  | Generate 20 variations per card |
| `cd client && npm run dev` | Start scanner app               |

## 📊 Training Details

-   **Model**: MobileNetV2 (transfer learning)
-   **Input Size**: 224x224 RGB
-   **Training Time**: ~5-10 min per card on Colab GPU
-   **Model Size**: ~9.4MB
-   **Accuracy**: 95%+ with 20+ augmentations per card

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

## 🎓 Tips for Best Results

1. **Start Small**: Test with 2 cards first
2. **Good Images**: Use high-quality card images (400x560+ px)
3. **More Augmentation**: Increase variations for better robustness
4. **Longer Training**: 50+ epochs for production models
5. **More Data**: 50+ real photos per card for ultimate accuracy

## 📄 License

ISC
