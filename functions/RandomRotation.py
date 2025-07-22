import cv2
import numpy as np
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RandomRotation:
    def __init__(self, angle_range=(-15, 15)):
        self.angle_range = angle_range

    def __call__(self, image):
        if isinstance(image, list):
            image = np.stack(image, axis=0)

        if isinstance(image, np.ndarray) and image.ndim == 4:
            rotated_images = []
            for i, img in enumerate(image):
                angle = random.uniform(self.angle_range[0], self.angle_range[1])
                logging.info(f"Image {i} rotated with angle: {angle:.2f} degrees")
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
                rotated_images.append(rotated_img)
            return np.stack(rotated_images, axis=0)

        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        logging.info(f"Single image rotated with angle: {angle:.2f} degrees")
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)