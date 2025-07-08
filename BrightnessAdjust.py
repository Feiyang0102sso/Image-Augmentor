import numpy as np
import random

class BrightnessAdjust:
    def __init__(self, delta_range=(-30, 30)):
        self.delta_range = delta_range

    def __call__(self, image):
        if isinstance(image, list):
            image = np.stack(image, axis=0)

        if isinstance(image, np.ndarray) and image.ndim == 4:
            return np.stack([self.__call__(img) for img in image], axis=0)

        delta = random.uniform(self.delta_range[0], self.delta_range[1])
        return np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)
