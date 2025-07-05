# single_demo.py

import cv2
from augmentor_module import ImageAugmentor
import os
import sys
import matplotlib.pyplot as plt
import json

def show_comparisons(original, augmented_list, titles):
    """
    original: 原图 (BGR)
    augmented_list: list of images (BGR)
    titles: list of str
    """
    # 将所有图像 BGR -> RGB
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    augmented_rgbs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in augmented_list]

    num_cols = len(augmented_list) + 1
    plt.figure(figsize=(5 * num_cols, 5))

    plt.subplot(1, num_cols, 1)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(original_rgb)

    for idx, (img, title) in enumerate(zip(augmented_rgbs, titles), start=2):
        plt.subplot(1, num_cols, idx)
        plt.title(title)
        plt.axis('off')
        plt.imshow(img)

    plt.tight_layout()
    plt.savefig("demo_picture/single_augmented_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    image_path = "../demo_picture/single_demo.JPEG"
    config_file = "../config.json"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到示例图片 '{image_path}'，请先准备好该图片后再运行。")

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"无法加载图片 '{image_path}'，请确保文件路径正确且文件未损坏。")

    # 加载 pipeline 配置
    with open(config_file, 'r') as f:
        config = json.load(f)

    pipeline = config.get('pipeline', [])

    augmentor = ImageAugmentor(config_path=config_file)

    augmented_list = []
    titles = []

    # 每次都基于原图进行单独增强
    for operation in pipeline:
        op_name = operation['name']
        params = operation.get('params', {})
        method = getattr(augmentor, f"_{op_name}")
        augmented_image = method(original_image.copy(), **params)
        augmented_list.append(augmented_image)
        # 设置展示标题
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        if param_str:
            titles.append(f"{op_name}({param_str})")
        else:
            titles.append(f"{op_name}")

    # 可视化全部结果
    show_comparisons(original_image, augmented_list, titles)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[错误] {e}")
        sys.exit(1)
