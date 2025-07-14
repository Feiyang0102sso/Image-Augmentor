import cv2
import numpy as np
import random
import logging

# Configure logging to record operations with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class RandomAffine:
    def __init__(self, max_translate=0.2):
        self.max_translate = max_translate

    def __call__(self, image):
        if isinstance(image, list):
            # Convert list of images to a stacked numpy array for batch processing
            image = np.stack(image, axis=0)

        if isinstance(image, np.ndarray) and image.ndim == 4:
            # Process each image in the batch recursively
            transformed_images = []
            for i, img in enumerate(image):
                h, w = img.shape[:2]
                tx = random.uniform(-self.max_translate, self.max_translate) * w
                ty = random.uniform(-self.max_translate, self.max_translate) * h
                logging.info(f"Image {i} translated with tx: {tx:.2f}, ty: {ty:.2f}")
                matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                transformed_img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
                transformed_images.append(transformed_img)
            return np.stack(transformed_images, axis=0)

        # Process single image
        h, w = image.shape[:2]
        tx = random.uniform(-self.max_translate, self.max_translate) * w
        ty = random.uniform(-self.max_translate, self.max_translate) * h
        logging.info(f"Single image translated with tx: {tx:.2f}, ty: {ty:.2f}")
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)