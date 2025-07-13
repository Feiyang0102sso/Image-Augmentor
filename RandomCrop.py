import cv2
import numpy as np
import random

class RandomCrop:
    def __init__(self, crop_size=(224, 224)):
        self.crop_size = crop_size

    def __call__(self, image):
        if isinstance(image, list):
            image = np.stack(image, axis=0)
        if isinstance(image, np.ndarray) and image.ndim == 4:
            return np.stack([self.__call__(img) for img in image], axis=0)

        h, w = image.shape[:2]
        ch, cw = self.crop_size
        if h < ch or w < cw:
            raise ValueError("Crop size should be smaller than image size.")
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        return image[top:top+ch, left:left+cw]
