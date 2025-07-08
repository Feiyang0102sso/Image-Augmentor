import cv2
import numpy as np
import random

class RandomRotation:
    def __init__(self, angle_range=(-15, 15)):
        self.angle_range = angle_range

    def __call__(self, image):
        if isinstance(image, list):
            image = np.stack(image, axis=0)

        if isinstance(image, np.ndarray) and image.ndim == 4:
            return np.stack([self.__call__(img) for img in image], axis=0)

        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
