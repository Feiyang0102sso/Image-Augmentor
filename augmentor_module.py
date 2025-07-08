import json
import importlib
import numpy as np

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
            try:
                module = importlib.import_module(op_name)
                cls = getattr(module, op_name)
                transforms.append(cls(**params))
            except ModuleNotFoundError:
                raise ValueError(f"未找到增强模块 {op_name}.py，请确认文件存在。")
            except AttributeError:
                raise ValueError(f"{op_name}.py 中未找到类 {op_name}，请确认实现正确。")
        return transforms

    def augment(self, images):
        if isinstance(images, list):
            images = np.stack(images, axis=0)

        if isinstance(images, np.ndarray) and images.ndim == 4:
            for transform in self.transforms:
                images = transform(images)
            return [images[i] for i in range(images.shape[0])]
        elif isinstance(images, np.ndarray) and images.ndim == 3:
            augmented_image = images.copy()
            for transform in self.transforms:
                augmented_image = transform(augmented_image)
            return augmented_image
        else:
            raise ValueError("输入类型不支持，请输入 np.ndarray[H,W,C] 或 np.ndarray[N,H,W,C] 或 List[np.ndarray]")
