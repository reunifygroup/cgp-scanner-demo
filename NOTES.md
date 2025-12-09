# Main Concerns

1. **GPU Resource Usage for traning**: Using web GPU services like Google Colab has a daily limit (~6H) and weekly limit (~10H). There is alternatives that leave a bit more time but still limited. Using local machine increase enourmously the waiting time.
2. **Testing waiting time optimization**: We are using 4 cards to reduce the training time, but this also means the there is much less data to train for. This cause the AI to be a lot less informed about what makes a card so.

# Solution 1 (TENSORFLOWJS + Google Colab)

-   Creating around 20 simple agumentations of the same original card image. This didn't work: the variantions were too simple. Result: AI recognized everything as 1-2 cards with very high confidence.
-   Creating around 50 higher quality augmentation with python (3D perspective, realistic lighting). Result: The cards are properly recnognized once the camera proportions is proper to the card object.

# Solution 2 (TENSORFLOWJS + Kaggle)

-   Google Colab single GPU and limits are really limiting. Kaggle free tier is much faster and higher free tier (30H per week). This solution won't ever scale properly with 20k+ cards.

# Known issue

-   The model WANTS always to recognize something in the camera. Even when no card is there, the model says a card is there and guess which one, giving always the same wrong result.

# Solution 3 (EMBEDDING BASED RECOGNITION ENGINE)

Problems with CNN Classification at Scale

1. Model size grows with cards: With 20k+ cards, your final classification layer alone would have millions of parameters (e.g., 512
   features × 20,000 classes = 10M+ parameters just for the output layer). This is enormous for in-browser use.
2. Retraining nightmare: Every time you add new cards, you need to retrain the entire model. With 20k+ cards, this becomes impractical.
3. Recognition unreliability: CNN classifiers struggle with subtle visual differences between similar cards. With thousands of
   similar-looking cards, the confusion increases exponentially.
4. Augmentation doesn't solve the core problem: You could generate 1000 augmentations per card and it still won't fix the fundamental issue

-   classification doesn't scale.

Why Embedding-Based Recognition is the Answer

Professional scanner apps (TCGPlayer, Delver Lens, etc.) use embedding-based similarity search, not classification. Here's how it works:

1. Fixed model size: The CNN creates a fixed-size feature vector (embedding) - e.g., 512 dimensions. Model stays the same whether you have
   10 cards or 1 million.
2. No retraining needed: Compute embeddings once for each reference card image. Add new cards by just computing their embeddings - no model
   updates.
3. Better accuracy: Uses similarity matching (find the most visually similar card) rather than trying to force the model to learn 20k
   different classes.
4. In-browser friendly: Small model (~5-20MB) + fast similarity search using libraries like FAISS or even simple cosine similarity.

Architecture Comparison

Current (Classification):
Image → CNN → [20k outputs] → Softmax → Card ID
❌ Model grows with cards
❌ Requires retraining
❌ Heavy for browser

Embedding (Similarity Search):
Image → CNN → [512-dim embedding] → Nearest Neighbor Search → Card ID
✅ Fixed model size
✅ No retraining
✅ Browser-friendly

Implementation Path

1. Model: Use a pre-trained feature extractor (MobileNetV3, EfficientNet) - remove classification head
2. Index: Pre-compute embeddings for all reference cards, store in JSON/binary format
3. Runtime: Extract embedding from camera → find nearest neighbor (cosine similarity)
4. Storage: ~2KB per card embedding (512 floats) = ~40MB for 20k cards

---

One suggestion: Your augmentations are quite gentle (which is good for classification). For embeddings, you could be slightly more
aggressive since we're doing similarity matching. But let's test with current settings first - they should work well.

---

`source .venv/bin/activate`

`pip freeze > requirements.txt`
`pip install -r requirements.txt`
