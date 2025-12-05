# Main Concerns

1. **GPU Resource Usage for traning**: Using web GPU services like Google Colab has a daily limit (~6H) and weekly limit (~10H). There is alternatives that leave a bit more time but still limited. Using local machine increase enourmously the waiting time.
2. **Testing waiting time optimization**: We are using 4 cards to reduce the training time, but this also means the there is much less data to train for. This cause the AI to be a lot less informed about what makes a card so.

# Solution 1 (TENSORFLOWJS)

-   Creating many agumentations of the same original card image. This didn't work: the variantions were too simple. Result: AI recognized everything as 1-2 cards with very high confidence.
-   Creating higher quality augmentation with python (3D perspective, realistic lighting). Result: ?
