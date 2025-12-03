import sharp from 'sharp';
import fs from 'fs/promises';

// 📐 Detect and extract card from image
export const extractCard = async (imagePath, outputPath) => {
  try {
    // Load image
    const image = sharp(imagePath);
    const metadata = await image.metadata();

    // 🔍 Simple approach: Crop center portion (assuming card is centered)
    // More advanced: Use edge detection, but sharp doesn't have built-in edge detection
    const cropWidth = Math.floor(metadata.width * 0.6);
    const cropHeight = Math.floor(metadata.height * 0.8);
    const left = Math.floor((metadata.width - cropWidth) / 2);
    const top = Math.floor((metadata.height - cropHeight) / 2);

    // Crop, resize to standard size, and enhance
    await image
      .extract({
        left,
        top,
        width: cropWidth,
        height: cropHeight
      })
      .resize(400, 560, {
        fit: 'fill',
        background: { r: 255, g: 255, b: 255, alpha: 1 }
      })
      .sharpen()
      .normalize() // Auto-adjust brightness/contrast
      .toFile(outputPath);

    return true;
  } catch (error) {
    console.error('Card extraction failed:', error);
    return false;
  }
};

// 🎨 Preprocess image for better hashing
export const preprocessForHash = async (imagePath, outputPath) => {
  try {
    await sharp(imagePath)
      .resize(400, 560, { fit: 'fill' })
      .normalize()
      .sharpen()
      .toFile(outputPath);

    return true;
  } catch (error) {
    console.error('Preprocessing failed:', error);
    return false;
  }
};
