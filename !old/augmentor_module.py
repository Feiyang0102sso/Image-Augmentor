import cv2
import numpy as np
import json
import random

class ImageAugmentor:

    def __init__(self, config_path):
        """
        Args:
            config_path: JSON配置文件路径，可选。
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.pipeline = self.config.get('pipeline', [])
        self._validate_pipeline()

    def _validate_pipeline(self):

        """验证pipeline中的操作是否都存在对应的实现方法。"""

        for operation in self.pipeline:
            op_name = operation.get('name')
            if not hasattr(self, f"_{op_name}"):
                raise ValueError(f"不支持的操作: {op_name}。请检查配置文件或模块实现。")

    def _random_horizontal_flip(self, image, p=0.5):

        """
        以给定的概率对图像进行随机水平翻转。
        """

        if random.random() < p:
            return cv2.flip(image, 1)
        return image

    def _random_rotation(self, image, angle_range=(-15, 15)):
        """
        在指定角度范围内对图像进行随机旋转。
        """
        angle = random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))

    def _brightness_adjust(self, image, delta_range=(-30, 30)):
        """
        随机调整图像的亮度。
        """
        delta = random.uniform(delta_range[0], delta_range[1])
        # 使用np.clip确保像素值在[0, 255]范围内
        return np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    def _contrast_adjust(self, image, factor_range=(0.8, 1.2)):
        """
        随机调整图像的对比度。
        """
        factor = random.uniform(factor_range[0], factor_range[1])
        # 使用np.clip确保像素值在[0, 255]范围内
        return np.clip(128 + factor * (image.astype(np.float32) - 128), 0, 255).astype(np.uint8)

    def augment(self, image):
        """
        单张增强
        """
        augmented_image = image.copy()
        for operation in self.pipeline:
            op_name = operation['name']
            params = operation.get('params', {})
            method = getattr(self, f"_{op_name}")
            augmented_image = method(augmented_image, **params)
        return augmented_image

    def augment_batch(self, images):
        """
        批量增强
        Return 包含所有增强后图像的列表。
        """
        return [self.augment(image) for image in images]