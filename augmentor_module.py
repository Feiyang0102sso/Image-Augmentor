import json
import importlib
import random
import logging

import numpy as np

from RandomChoice import RandomChoice


class ImageAugmentor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.pipeline = self.config.get('pipeline', [])
        self.transforms = self._build_pipeline()

    def _build_pipeline(self):
        transforms = []
        for operation in self.pipeline:
            op_name = operation['name']
            params = operation.get('params', {})
            prob = operation.get('prob', 1.0)
            try:
                if op_name == "RandomChoice":
                    choices = []
                    for choice in params['choices']:
                        choice_name = choice['name']
                        choice_params = choice.get('params', {})
                        module = importlib.import_module(choice_name)
                        cls = getattr(module, choice_name)
                        choices.append(cls(**choice_params))
                    transforms.append(RandomChoice(choices))
                    logging.info(f"Building RandomChoice with: {[c.__class__.__name__ for c in choices]}")
                else:
                    module = importlib.import_module(op_name)
                    cls = getattr(module, op_name)
                    transform = cls(**params)

                    # wrap with prob
                    if prob < 1.0:
                        transform = self._wrap_with_probability(transform, prob, op_name)
                    transforms.append(transform)
            except ModuleNotFoundError:
                raise ValueError(f"未找到增强模块 {op_name}.py，请确认文件存在。")
            except AttributeError:
                raise ValueError(f"{op_name}.py 中未找到类 {op_name}，请确认实现正确。")
        return transforms

    def _wrap_with_probability(self, transform, prob, transform_name):
        def wrapped(image):
            if random.random() < prob:
                result = transform(image)
                logging.info(f"Applied {transform_name} with probability {prob}")
                return result
            else:
                logging.info(f"Skipped {transform_name} with probability {prob}")
                return image

        return wrapped

    def augment(self, images):
        # 堆叠图像
        if isinstance(images, list):
            images = np.stack(images, axis=0)

        # 批处理
        if isinstance(images, np.ndarray) and images.ndim == 4:
            for transform in self.transforms:
                images = transform(images)
            return [images[i] for i in range(images.shape[0])]

        # 单张处理
        elif isinstance(images, np.ndarray) and images.ndim == 3:
            augmented_image = images.copy()
            for transform in self.transforms:
                augmented_image = transform(augmented_image)
            return augmented_image
        else:
            raise ValueError("输入类型不支持，请输入 np.ndarray[H,W,C] 或 np.ndarray[N,H,W,C] 或 List[np.ndarray]")