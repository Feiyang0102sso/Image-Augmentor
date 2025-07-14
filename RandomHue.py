import cv2
import numpy as np
import random
import logging

# Configure logging to record operations with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class RandomHue:
    def __init__(self, delta=18):
        self.delta = delta

    def __call__(self, image):
        if isinstance(image, list):
            # Convert list of images to a stacked numpy array for batch processing
            image = np.stack(image, axis=0)

        if isinstance(image, np.ndarray) and image.ndim == 4:
            # Process each image in the batch recursively
            adjusted_images = []
            for i, img in enumerate(image):
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
                hue_shift = random.randint(-self.delta, self.delta)
                logging.info(f"Image {i} adjusted with hue shift: {hue_shift}")
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                adjusted_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                adjusted_images.append(adjusted_img)
            return np.stack(adjusted_images, axis=0)

        # Process single image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
        hue_shift = random.randint(-self.delta, self.delta)
        logging.info(f"Single image adjusted with hue shift: {hue_shift}")
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)