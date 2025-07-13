import cv2
import numpy as np
import random

class RandomHue:
    def __init__(self, delta=18):
        self.delta = delta

    def __call__(self, image):
        if isinstance(image, list):
            image = np.stack(image, axis=0)
        if isinstance(image, np.ndarray) and image.ndim == 4:
            return np.stack([self.__call__(img) for img in image], axis=0)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
        hue_shift = random.randint(-self.delta, self.delta)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
