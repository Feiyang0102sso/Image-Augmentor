import cv2
import numpy as np
import random
import logging

# Configure logging to record operations with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RandomCrop:
    def __init__(self, crop_size=(224, 224)):
        self.crop_size = crop_size

    def __call__(self, image):
        if isinstance(image, list):
            # Convert list of images to a stacked numpy array for batch processing
            image = np.stack(image, axis=0)

        if isinstance(image, np.ndarray) and image.ndim == 4:
            # Process each image in the batch recursively
            cropped_images = []
            for i, img in enumerate(image):
                h, w = img.shape[:2]
                ch, cw = self.crop_size
                if h < ch or w < cw:
                    raise ValueError("Crop size should be smaller than image size.")
                top = random.randint(0, h - ch)
                left = random.randint(0, w - cw)
                logging.info(f"Image {i} cropped at top: {top}, left: {left} with size {ch}x{cw}")
                cropped_img = img[top:top + ch, left:left + cw]
                cropped_images.append(cropped_img)
            return np.stack(cropped_images, axis=0)

        # Process single image
        h, w = image.shape[:2]
        ch, cw = self.crop_size
        if h < ch or w < cw:
            raise ValueError("Crop size should be smaller than image size.")
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        logging.info(f"Single image cropped at top: {top}, left: {left} with size {ch}x{cw}")
        return image[top:top + ch, left:left + cw]