"""
# 🎴 Pokémon Card Classifier Training
# Run this in Kaggle with GPU enabled

## Setup Instructions:
1. Upload training-data.zip to Colab using files.upload()
2. Unzip it: !unzip training-data.zip
3. Copy and run this entire script
4. Model will auto-download at the end
"""

# ============================================================================
# 📦 STEP 1: Install Dependencies
# ============================================================================

!pip install -q tensorflowjs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import json
import tensorflowjs as tfjs
import glob

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# ============================================================================
# 📁 STEP 2: Configure Data Paths
# ============================================================================

TRAINING_DATA_PATH = '/kaggle/input/training-data' # Change to your dataset name

# Local paths in Kaggle
TRAIN_DIR = '/kaggle/working/cgpremium/scanner/train'
VAL_DIR = '/kaggle/working/cgpremium/scanner/val'
MODEL_OUTPUT_DIR = '/kaggle/working/cgpremium/scanner/model_output'

# Hyperparameters - OPTIMIZED for advanced pre-augmented data
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Can use larger batches with pre-augmented data
EPOCHS = 100  # Fewer epochs needed with high-quality augmentation
LEARNING_RATE = 0.0005  # Can use higher LR with pre-augmented data

print(f"✅ Using training data from: {TRAINING_DATA_PATH}")

# ============================================================================
# 🧹 STEP 2.5: Clean Up Previous Run
# ============================================================================

print("\n🧹 Cleaning up previous training artifacts...")

cleanup_paths = [
    TRAIN_DIR,
    VAL_DIR,
    MODEL_OUTPUT_DIR,
    '/kaggle/working/cgpremium/scanner/saved_model',
    '/kaggle/working/cgpremium/scanner/model.keras',
    '/kaggle/working/cgpremium/scanner/best_model.keras',
    '/kaggle/working/cgpremium/scanner/training_history.png',
]

# Remove directories and files
for path in cleanup_paths:
    try:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"   🗑️  Removed directory: {os.path.basename(path)}")
            else:
                os.remove(path)
                print(f"   🗑️  Removed file: {os.path.basename(path)}")
    except Exception as e:
        print(f"   ⚠️  Could not remove {path}: {e}")

# Remove any .zip files in the directory
for zip_file in glob.glob('/kaggle/working/cgpremium/scanner/*.zip'):
    try:
        os.remove(zip_file)
        print(f"   🗑️  Removed zip: {os.path.basename(zip_file)}")
    except Exception as e:
        print(f"   ⚠️  Could not remove {zip_file}: {e}")

print("✅ Cleanup complete - ready for fresh training!\n")

# ============================================================================
# 🔄 STEP 3: Prepare Dataset (Train/Val Split)
# ============================================================================

def create_train_val_split(source_dir, train_dir, val_dir, val_split=0.2):
    """Split data into train and validation sets"""

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    total_train = 0
    total_val = 0
    all_card_classes = []

    # Get all images directly in source_dir
    all_images = [f for f in os.listdir(source_dir)
                  if f.endswith(('.png', '.jpg')) and os.path.isfile(os.path.join(source_dir, f))]

    if not all_images:
        raise Exception(f"No images found in {source_dir}! Please check your training data.")

    print(f"📦 Found {len(all_images)} images\n")

    # Group images by card ID (extracted from filename)
    card_groups = {}
    for img in all_images:
        # Extract card ID from filename
        # For "cardname_aug0.png" -> "cardname"
        # For "cardname_original.png" -> "cardname"
        parts = img.split('_')
        if parts[-1].startswith('aug') or parts[-1] == 'original.png':
            card_id = '_'.join(parts[:-1])
        else:
            card_id = img.rsplit('.', 1)[0]  # Remove extension

        if card_id not in card_groups:
            card_groups[card_id] = []
        card_groups[card_id].append(img)

    print(f"📋 Detected {len(card_groups)} unique cards\n")

    # Process each card
    for card_id, card_images in card_groups.items():
        all_card_classes.append(card_id)

        # Split images
        np.random.shuffle(card_images)
        val_count = max(1, int(len(card_images) * val_split))
        train_count = len(card_images) - val_count

        train_images = card_images[:train_count]
        val_images = card_images[train_count:]

        # Create class directories
        train_class_dir = os.path.join(train_dir, card_id)
        val_class_dir = os.path.join(val_dir, card_id)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Copy files
        for img in train_images:
            shutil.copy2(
                os.path.join(source_dir, img),
                os.path.join(train_class_dir, img)
            )
            total_train += 1

        for img in val_images:
            shutil.copy2(
                os.path.join(source_dir, img),
                os.path.join(val_class_dir, img)
            )
            total_val += 1

        print(f"✅ {card_id}: {train_count} train, {val_count} val")

    print(f"\n📊 Total: {total_train} train, {total_val} val images")
    return total_train, total_val, sorted(all_card_classes)

# Create split
total_train, total_val, class_names_list = create_train_val_split(
    TRAINING_DATA_PATH, TRAIN_DIR, VAL_DIR, val_split=0.2
)

# ============================================================================
# 🎨 STEP 4: Data Generators with LIGHT Augmentation
# ============================================================================

# Light augmentation since pre-augmentation already did heavy lifting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,  # Light rotation for extra variety
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.85, 1.15],  # Subtle brightness only
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Validation with only rescaling (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"\n📦 Number of card classes: {num_classes}")
print(f"📋 Classes: {list(train_generator.class_indices.keys())}")

# ============================================================================
# 🧠 STEP 5: Build SMALLER CNN Model (prevents overfitting)
# ============================================================================

# Clear Keras layer name counter for consistent naming
keras.backend.clear_session()

def create_model(num_classes, img_size=IMG_SIZE):
    """Create lightweight CNN classifier optimized for small datasets"""

    # Smaller model to prevent overfitting on small dataset
    model = keras.Sequential([
        # Conv Block 1
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(*img_size, 3)),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Conv Block 2
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Conv Block 3
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.4),

        # Classification head - smaller to prevent overfitting
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),  # Reduced from 256
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

model = create_model(num_classes)

# Explicitly build the model with input shape
model.build((None, *IMG_SIZE, 3))

print("\n📋 Model layer names:")
for i, layer in enumerate(model.layers):
    print(f"   {i}: {layer.name} ({layer.__class__.__name__})")

model.summary()

# ============================================================================
# ⚙️ STEP 6: Compile Model
# ============================================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

# ============================================================================
# 🎯 STEP 7: Callbacks
# ============================================================================

callbacks = [
    # Reduce learning rate when validation loss plateaus
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,  # Increased patience
        verbose=1,
        min_lr=1e-7
    ),

    # Early stopping with more patience
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,  # Increased from 10
        restore_best_weights=True,
        verbose=1
    ),

    # Model checkpoint
    keras.callbacks.ModelCheckpoint(
        '/kaggle/working/cgpremium/scanner/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ============================================================================
# 🚀 STEP 8: Train Model
# ============================================================================

print("\n🚀 Starting training...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training complete!")

# ============================================================================
# 📊 STEP 9: Evaluate Model
# ============================================================================

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('/kaggle/working/cgpremium/scanner/training_history.png')
plt.show()

# Final evaluation
val_loss, val_accuracy, val_top_k = model.evaluate(val_generator)
print(f"\n📊 Final Validation Results:")
print(f"   Loss: {val_loss:.4f}")
print(f"   Accuracy: {val_accuracy*100:.2f}%")
print(f"   Top-3 Accuracy: {val_top_k*100:.2f}%")

# ============================================================================
# 💾 STEP 10: Export to TensorFlow.js
# ============================================================================

# Create output directory
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Save class names
print(f"\n📋 Saving class names: {class_names_list}")
with open(os.path.join(MODEL_OUTPUT_DIR, 'class_names.json'), 'w') as f:
    json.dump(class_names_list, f, indent=2)

# Convert to TensorFlow.js format (Graph Model - better Keras 3.x compatibility)
print("\n📦 Converting to TensorFlow.js format...")

# Step 1: Save as TensorFlow SavedModel
SAVED_MODEL_PATH = '/kaggle/working/cgpremium/scanner/saved_model'
print(f"Saving as TensorFlow SavedModel...")
model.export(SAVED_MODEL_PATH)

# Step 2: Convert to TensorFlow.js Graph Model format
print(f"Converting SavedModel to TensorFlow.js Graph Model...")
tfjs.converters.convert_tf_saved_model(
    SAVED_MODEL_PATH,
    MODEL_OUTPUT_DIR
)

print(f"✅ Model exported to: {MODEL_OUTPUT_DIR}")

# Check what files were created
print("\n📂 Files in model output directory:")
!ls -lh {MODEL_OUTPUT_DIR}

print("\n✅ Graph Model ready for TensorFlow.js!")

# Create a zip file
print("\n📦 Creating zip file...")
shutil.make_archive('/kaggle/working/cgpremium/scanner/card_classifier_model', 'zip', MODEL_OUTPUT_DIR)
print("✅ Model packaged as: /kaggle/working/cgpremium/scanner/card_classifier_model.zip")

# Download model
print("\n⬇️  Downloading model...")
# Model saved to /kaggle/working/cgpremium/scanner/card_classifier_model.zip
# Download from Output panel →
print("\n✅ Model saved! Check Output panel to download:")
print("   /kaggle/working/cgpremium/scanner/card_classifier_model.zip")

print("\n" + "="*50)
print("🎉 TRAINING COMPLETE!")
print("="*50)
print("\n📋 Next Steps:")
print("1. Extract card_classifier_model.zip")
print("2. Place files in your project's client/public/model/ directory")
print("3. Load model in React app with TensorFlow.js")
print("\n📊 Model Details:")
print(f"- Classes: {num_classes} cards")
print(f"- Val Accuracy: {val_accuracy*100:.2f}%")
print(f"- Input size: 224x224")
print("\n⚠️  IMPORTANT: For production use, you need:")
print("   - At least 10-20 REAL photos per card (different angles/lighting)")
print("   - Not just augmentations of 1 image per card")
print("   - This will dramatically improve real-world accuracy")
