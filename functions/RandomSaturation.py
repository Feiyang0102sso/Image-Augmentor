import cv2
import numpy as np
import random
import logging

# Configure logging to record operations with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class RandomSaturation:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, image):
        if isinstance(image, list):
            # Convert list of images to a stacked numpy array for batch processing
            image = np.stack(image, axis=0)

        if isinstance(image, np.ndarray) and image.ndim == 4:
            # Process each image in the batch recursively
            adjusted_images = []
            for i, img in enumerate(image):
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                saturation_scale = random.uniform(self.lower, self.upper)
                logging.info(f"Image {i} adjusted with saturation scale: {saturation_scale:.2f}")
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
                adjusted_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                adjusted_images.append(adjusted_img)
            return np.stack(adjusted_images, axis=0)

        # Process single image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation_scale = random.uniform(self.lower, self.upper)
        logging.info(f"Single image adjusted with saturation scale: {saturation_scale:.2f}")
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)