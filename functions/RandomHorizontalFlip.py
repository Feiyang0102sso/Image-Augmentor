import cv2
import numpy as np
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if isinstance(image, list):
            image = np.stack(image, axis=0)

        if isinstance(image, np.ndarray) and image.ndim == 4:
            # batch
            flipped_images = []
            for i, img in enumerate(image):
                if random.random() < self.p:
                    logging.info(f"Image {i} flipped horizontally with probability {self.p}")
                    flipped_img = cv2.flip(img, 1)
                else:
                    logging.info(f"Image {i} not flipped, probability {self.p}")
                    flipped_img = img
                flipped_images.append(flipped_img)
            return np.stack(flipped_images, axis=0)

        # single
        if random.random() < self.p:
            logging.info(f"Single image flipped horizontally with probability {self.p}")
            return cv2.flip(image, 1)
        logging.info(f"Single image not flipped, probability {self.p}")
        return image