import numpy as np
import random
import logging

# Configure logging to record operations with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ContrastAdjust:
    def __init__(self, factor_range=(0.8, 1.2)):
        self.factor_range = factor_range

    def __call__(self, image):

        if isinstance(image, list):
            # Convert list of images to a stacked numpy array for batch processing
            image = np.stack(image, axis=0)

        if isinstance(image, np.ndarray) and image.ndim == 4:
            # Process each image in the batch recursively
            adjusted_images = []
            for i, img in enumerate(image):
                factor = random.uniform(self.factor_range[0], self.factor_range[1])
                logging.info(f"Image {i} adjusted with contrast factor: {factor:.2f}")
                adjusted_img = np.clip(128 + factor * (img.astype(np.float32) - 128), 0, 255).astype(np.uint8)
                adjusted_images.append(adjusted_img)
            return np.stack(adjusted_images, axis=0)

        # Process single image
        factor = random.uniform(self.factor_range[0], self.factor_range[1])
        logging.info(f"Single image adjusted with contrast factor: {factor:.2f}")
        return np.clip(128 + factor * (image.astype(np.float32) - 128), 0, 255).astype(np.uint8)