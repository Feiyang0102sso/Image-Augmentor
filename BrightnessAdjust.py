import numpy as np
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BrightnessAdjust:
    def __init__(self, delta_range=(-30, 30)):
        self.delta_range = delta_range

    def __call__(self, image):
        if isinstance(image, list):
            image = np.stack(image, axis=0)

        if isinstance(image, np.ndarray) and image.ndim == 4:
            adjusted_images = []
            for i, img in enumerate(image):
                delta = random.uniform(self.delta_range[0], self.delta_range[1])
                logging.info(f"Image {i} adjusted with brightness delta: {delta:.2f}")
                adjusted_img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
                adjusted_images.append(adjusted_img)
            return np.stack(adjusted_images, axis=0)

        delta = random.uniform(self.delta_range[0], self.delta_range[1])
        logging.info(f"Single image adjusted with brightness delta: {delta:.2f}")
        return np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)