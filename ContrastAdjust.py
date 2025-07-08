import numpy as np
import random

class ContrastAdjust:
    def __init__(self, factor_range=(0.8, 1.2)):
        self.factor_range = factor_range

    def __call__(self, image):
        if isinstance(image, list):
            image = np.stack(image, axis=0)

        if isinstance(image, np.ndarray) and image.ndim == 4:
            return np.stack([self.__call__(img) for img in image], axis=0)

        factor = random.uniform(self.factor_range[0], self.factor_range[1])
        return np.clip(128 + factor * (image.astype(np.float32) - 128), 0, 255).astype(np.uint8)
