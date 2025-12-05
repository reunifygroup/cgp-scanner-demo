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

### 5️⃣ Deploy Model

```bash
# Extract model files
unzip card_classifier_model.zip

# Move to client
mv model.json group1-shard1of1.bin class_names.json client/public/model/

# Start app
cd client && npm run dev
```

## 📊 Expected Results

| Cards  | Training Time | Expected Accuracy | Use Case         |
| ------ | ------------- | ----------------- | ---------------- |
| 4      | ~2 min        | 60-75%            | Pipeline testing |
| 10     | ~5 min        | 70-80%            | Quick demo       |
| 100    | ~20 min       | 80-90%            | Prototype        |
| 1,000  | ~3 hours      | 90-95%            | Production beta  |
| 20,000 | ~60 hours     | 95%+              | Pro-level app    |

## 🎓 Why This Works

### Basic Augmentation (Old Approach)

-   Simple rotation, brightness
-   2D transformations only
-   **Result**: Model memorizes flat images, fails on real camera views

### Advanced Augmentation (New Approach)

-   3D perspective transforms
-   Realistic lighting simulation
-   Camera effect simulation
-   **Result**: Model learns card features, works like pro apps

### Scale Effect

With **4 cards**: Limited data, harder to generalize
With **100+ cards**: Enough data to learn general patterns
With **20,000 cards**: Production-quality like TCGPlayer

## 🚨 Important Notes

1. **Always use `augment-advanced`** - basic augmentation is outdated
2. **Test with 4 cards first** - validate pipeline works
3. **Scale gradually** - 4 → 10 → 100 → 1,000 → 20,000
4. **Monitor GPU usage** - Google Colab has limits
5. **Save checkpoints** - Download model after each training

## 🔄 Iterating

When adding more cards:

1. Add images to `images/` directory
2. Run `augment-advanced` again (overwrites `training-data/`)
3. Create new zip
4. Train in Colab
5. Deploy new model

## 💡 Pro Tips

-   **Start small** (4 cards) to validate pipeline
-   **Add cards gradually** - easier to debug issues
-   **Keep original images** - you can re-augment anytime
-   **Version your models** - use timestamps in filenames
-   **Test thoroughly** - try different lighting/angles with real camera

## 📈 Scaling to 20,000 Cards

Once your pipeline works with 4 cards:

1. **Download all card images** from TCGdex API
2. **Organize by set** (as you're doing now)
3. **Run `augment-advanced`** (will take ~2-3 hours)
4. **Train in Colab** (will take ~60 hours with GPU)
5. **Deploy** production model

**Storage needs**:

-   20,000 cards × 50 augmentations = 1M images
-   ~200GB of training data
-   ~50MB final model size

This is exactly how the big apps do it!
