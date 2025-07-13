import cv2
import numpy as np
import random

class RandomAffine:
    def __init__(self, max_translate=0.2):
        self.max_translate = max_translate

    def __call__(self, image):
        if isinstance(image, list):
            image = np.stack(image, axis=0)
        if isinstance(image, np.ndarray) and image.ndim == 4:
            return np.stack([self.__call__(img) for img in image], axis=0)

        h, w = image.shape[:2]
        tx = random.uniform(-self.max_translate, self.max_translate) * w
        ty = random.uniform(-self.max_translate, self.max_translate) * h

        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
