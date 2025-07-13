import cv2
import numpy as np
import random

class RandomSaturation:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, image):
        if isinstance(image, list):
            image = np.stack(image, axis=0)
        if isinstance(image, np.ndarray) and image.ndim == 4:
            return np.stack([self.__call__(img) for img in image], axis=0)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation_scale = random.uniform(self.lower, self.upper)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
