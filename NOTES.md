# Main Concerns

1. **GPU Resource Usage for traning**: Using web GPU services like Google Colab has a daily limit (~6H) and weekly limit (~10H). There is alternatives that leave a bit more time but still limited. Using local machine increase enourmously the waiting time.
2. **Testing waiting time optimization**: We are using 4 cards to reduce the training time, but this also means the there is much less data to train for. This cause the AI to be a lot less informed about what makes a card so.

# Solution 1 (TENSORFLOWJS)

-   Creating many agumentations of the same original card image. This didn't work: the variantions were too simple. Result: AI recognized everything as 1-2 cards with very high confidence.
-   Creating higher quality augmentation with python (3D perspective, realistic lighting). Result: The cards are properly recnognized once the camera proportions is proper to the card object.

# Known issue

-   The model WANTS always to recognize something in the camera. Even when no card is there, the model says a card is there and guess which one, giving always the same wrong result.

## Proposed solutions:

1.  Stricter Thresholds (Easy - No Retraining)

        A) Higher confidence + Gap check:
        const top1 = predictions[maxIndex]
        const top2 = predictions[secondMaxIndex]

        // Only show if:
        // - Top confidence > 70% AND
        // - Big gap between 1st and 2nd place
        if (top1 > 0.70 && (top1 - top2) > 0.25) {
        showCard()
        }

        Pros: Quick fix, no retraining
        Cons: Might miss some real cards at angles

2.  Add "Background" Class ⭐ (Best - Requires Retraining)

        Add a 5th class called "no_card":

        Training data:

        -   4 card classes (51 images each)
        -   1 background class (200 images: tables, hands, walls, random objects)

        Pros:

        -   Model learns what "not a card" looks like
        -   More robust long-term solution
        -   Scales well to 1000+ cards

        Cons:

        -   Need to collect ~200 background images
        -   Retrain model (~2 min)

        3. Entropy Threshold (Medium - No Retraining)

        Check if model is "confused":
        // If all predictions are similar, model is uncertain
        const entropy = calculateEntropy(predictions)
        if (entropy > threshold) {
        // Too uncertain - don't show anything
        }

        Pros: Catches when model doesn't know
        Cons: Need to tune threshold

3.  Object Detection First (Advanced - Major Change)

        Two-stage approach:

        1. Stage 1: Detect IF card exists (YOLO/SSD)
        2. Stage 2: Classify WHICH card (your CNN)

        Pros: Production-quality solution
        Cons: Complex, requires different architecture
