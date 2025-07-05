# multi_demo.py

import cv2
from augmentor_module import ImageAugmentor
import os
import sys
import matplotlib.pyplot as plt
import json

def show_batch_comparisons_grid(image_list, augmented_lists, titles_list, save_path):
    """
    显示多张图片，每张图对应横向排列：原图 + pipeline 增强后的图
    多张图片纵向排列，最终合并为一张图保存。
    """
    num_images = len(image_list)
    num_cols = len(augmented_lists[0]) + 1  # 原图 + pipeline 步骤数

    plt.figure(figsize=(5 * num_cols, 5 * num_images))

    for row_idx, (original, augmented_list, titles) in enumerate(zip(image_list, augmented_lists, titles_list)):
        # 转 RGB
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        augmented_rgbs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in augmented_list]

        # 第一列显示原图
        plt.subplot(num_images, num_cols, row_idx * num_cols + 1)
        plt.title("Original")
        plt.axis('off')
        plt.imshow(original_rgb)

        # 后续列显示增强后的图
        for col_idx, (aug_img, title) in enumerate(zip(augmented_rgbs, titles)):
            plt.subplot(num_images, num_cols, row_idx * num_cols + col_idx + 2)
            plt.title(title)
            plt.axis('off')
            plt.imshow(aug_img)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"[完成] 已保存批量增强对比图到 {save_path}")

def main():
    input_folder = "demo_picture/batchs"
    config_file = "../config.json"
    save_path = "../demo_picture/batchs/augmented_comparison_grid.png"

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"未找到输入文件夹 '{input_folder}'，请先准备好后再运行。")

    with open(config_file, 'r') as f:
        config = json.load(f)

    pipeline = config.get('pipeline', [])
    augmentor = ImageAugmentor(config_path=config_file)

    supported_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    generated_filename = "augmented_comparison_grid.png"
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in supported_ext and f != generated_filename
    ]

    if not image_files:
        print(f"[提示] 文件夹 '{input_folder}' 中未找到支持的图片文件。")
        return

    image_list = []
    augmented_lists = []
    titles_list = []

    for idx, image_name in enumerate(image_files, 1):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[警告] 跳过无法加载的图片: {image_name}")
            continue

        augmented_list = []
        titles = []

        # 对每张图片每步独立增强
        for operation in pipeline:
            op_name = operation['name']
            params = operation.get('params', {})
            method = getattr(augmentor, f"_{op_name}")
            augmented_image = method(image.copy(), **params)
            augmented_list.append(augmented_image)

            # 构造标题
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            if param_str:
                titles.append(f"{op_name}({param_str})")
            else:
                titles.append(f"{op_name}")

        image_list.append(image)
        augmented_lists.append(augmented_list)
        titles_list.append(titles)

        print(f"[{idx}/{len(image_files)}] 已处理: {image_name}")

    # 批量展示并保存
    show_batch_comparisons_grid(image_list, augmented_lists, titles_list, save_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[错误] {e}")
        sys.exit(1)
