# 🚀 Production Card Scanner Workflow

## Overview

This workflow creates a **production-quality card recognition system** that scales from 4 cards (testing) to 20,000+ cards (production) - just like TCGPlayer, Delver Lens, and other pro apps.

## Key Insight

Professional card recognition apps **don't photograph each card**. They use:

1. Official card images (from APIs)
2. **Advanced augmentation** to simulate real camera conditions
3. CNN classification (scales to thousands of cards)

## 🎯 Complete Workflow

### 1️⃣ Prepare Card Images

Place official card images in `images/`:

```
images/
├── sv09-001_Caterpie.png
├── sv09-003_Butterfree.png
├── sv10-001_Ethan_s_Pinsir.png
└── sv10-133_Scrafty.png
```

**Start with 4 cards for testing**, then scale to hundreds/thousands.

### 2️⃣ Generate Advanced Augmentation

```bash
# Install dependencies (first time only)
pip3 install -r requirements.txt

# Run advanced augmentation
npm run augment-advanced
```

This creates **50 high-quality variations per card** with:

-   ✅ 3D perspective transforms (card at angles)
-   ✅ Realistic lighting & shadows
-   ✅ Camera blur, noise, compression
-   ✅ Partial occlusion (fingers, glare)
-   ✅ Background integration

**Output**: `training-data/` with ~200 images for 4 cards

### 3️⃣ Create Training ZIP

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cd training-data && zip -r ../training-data_${TIMESTAMP}.zip . && cd ..
```

### 4️⃣ Train Model in Google Colab

1. **Open**: https://colab.research.google.com/
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Create working directory**:

    ```python
    !mkdir -p /content/cgpremium/scanner
    %cd /content/cgpremium/scanner
    ```

4. **Upload & Extract**:

    ```python
    from google.colab import files
    import glob
    import os

    # Upload zip
    uploaded = files.upload()

    # Find and extract
    zip_file = max(glob.glob('*.zip'), key=os.path.getctime)
    !rm -rf training-data
    !unzip -q {zip_file} -d training-data
    !rm {zip_file}
    ```

5. **Copy entire `train_card_classifier.py`** into a cell
6. **Run the cell** - training starts automatically
7. **Download** `card_classifier_model.zip` when done

### 4️⃣ (Alternative) Train Model in Kaggle Notebooks

**⭐ Recommended** - More free GPU hours (30/week vs ~12-15/week on Colab)

1. **Sign up/Login**: https://www.kaggle.com/

2. **Create new notebook**:

    - Click "Code" → "New Notebook"
    - Settings → Accelerator → **GPU T4 x2**
    - Settings → Internet → **On**

3. **Upload training data as dataset**:

    - Click "+ Add Data" → "Upload"
    - Select your `training-data_TIMESTAMP.zip`
    - Kaggle will extract it automatically to `/kaggle/input/your-dataset-name/`

4. **Setup** (first cell):

    ```python
    import os

    # Create working directory
    !mkdir -p /kaggle/working/cgpremium/scanner
    os.chdir('/kaggle/working/cgpremium/scanner')

    # Find your uploaded dataset (check "Data" tab for exact name)
    input_path = '/kaggle/input'
    dataset_folders = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]

    if dataset_folders:
        dataset_name = dataset_folders[0]
        print(f"✅ Found dataset: {dataset_name}")
        print(f"   Path: /kaggle/input/{dataset_name}")
    else:
        print("❌ No dataset found! Please upload your training-data zip")

    print(f"✅ Working directory: {os.getcwd()}")
    ```

5. **Install dependencies** (second cell):

    ```python
    !pip install -q tensorflowjs
    ```

    ```python
    import tensorflowjs as tfjs
    print(f"✅ TensorFlow.js version: {tfjs.__version__}")
    ```

6. **Training script** (third cell):

    - Copy entire `train_card_classifier.py` content
    - **Change ONLY the first path** at the top (use your dataset name from step 4):
        ```python
        TRAINING_DATA_PATH = '/kaggle/input/YOUR-DATASET-NAME'  # <-- Change this!
        TRAIN_DIR = '/kaggle/working/cgpremium/scanner/train'
        VAL_DIR = '/kaggle/working/cgpremium/scanner/val'
        MODEL_OUTPUT_DIR = '/kaggle/working/cgpremium/scanner/model_output'
        ```
    - At the end, **replace** the `files.download()` line with:
        ```python
        # Model saved to /kaggle/working/cgpremium/scanner/card_classifier_model.zip
        # Download from Output panel →
        print("\n✅ Model saved! Check Output panel to download:")
        print("   /kaggle/working/cgpremium/scanner/card_classifier_model.zip")
        ```
    - Run the cell!

    **⚡ Note**: Training reads directly from `/kaggle/input/`

7. **Download model** (after training completes):
    - Look in the **Output** panel on the right side
    - Find `card_classifier_model.zip`
    - Click the three dots → Download

## 🔄 Iterating

When adding more cards:

1. Add images to `images/` directory
2. Run `augment-advanced` again (overwrites `training-data/`)
3. Create new zip
4. Train in Colab
5. Deploy new model
