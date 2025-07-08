import cv2
import numpy as np
import random

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if isinstance(image, list):
            image = np.stack(image, axis=0)

        if isinstance(image, np.ndarray) and image.ndim == 4:
            # 批量处理
            return np.stack([self.__call__(img) for img in image], axis=0)

        # 单张处理
        if random.random() < self.p:
            return cv2.flip(image, 1)
        return image
