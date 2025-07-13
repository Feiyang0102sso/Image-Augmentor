import random
import logging

class RandomChoice:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        transform = random.choice(self.transforms)
        logging.info(f"[RandomChoice] Selected transform: {transform.__class__.__name__}")
        return transform(image)
