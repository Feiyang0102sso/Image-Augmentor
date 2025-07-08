import cv2
from augmentor_module import ImageAugmentor
import os
import sys
import json
import matplotlib.pyplot as plt
import logging

# === 可快速配置区域 ===
SINGLE_IMAGE_PATH = "demo_picture/single_demo.JPEG"
BATCH_FOLDER = "demo_picture/batchs"
CONFIG_FILE = "config.json"
SINGLE_SAVE_PATH = "demo_picture/single_augmented_comparison.png"
BATCH_SAVE_PATH = "demo_picture/batchs/augmented_comparison_grid.png"
SUPPORTED_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

def load_pipeline_from_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    pipeline = config.get("pipeline", [])
    if not pipeline:
        logging.error("配置文件中未找到 pipeline 或 pipeline 为空")
        sys.exit(1)
    return pipeline

def apply_all_operations(image, pipeline):
    from importlib import import_module

    augmented_images = []
    titles = []

    for op in pipeline:
        name = op["name"]
        params = op.get("params", {})

        try:
            module = import_module(name)
            cls = getattr(module, name)
            augmenter = cls(**params)
            augmented_image = augmenter(image.copy())
            augmented_images.append(augmented_image)
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            titles.append(f"{name}({param_str})" if param_str else name)
        except Exception as e:
            logging.warning(f"增强 {name} 时出错: {e}")

    return augmented_images, titles

def plot_all_grid(images, augmented_images_list, titles_list, save_path):
    num_images = len(images)
    num_cols = len(augmented_images_list[0]) + 1  # 原图 + 所有增强

    plt.figure(figsize=(5 * num_cols, 5 * num_images))

    for row_idx, (orig_img, aug_imgs, titles) in enumerate(zip(images, augmented_images_list, titles_list)):
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        plt.subplot(num_images, num_cols, row_idx * num_cols + 1)
        plt.title("Original")
        plt.axis('off')
        plt.imshow(orig_rgb)

        for col_idx, (aug_img, title) in enumerate(zip(aug_imgs, titles)):
            aug_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
            plt.subplot(num_images, num_cols, row_idx * num_cols + col_idx + 2)
            plt.title(title)
            plt.axis('off')
            plt.imshow(aug_rgb)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    logging.info(f"[批量] 已保存可视化结果到 {save_path}")

def run_single_demo():
    logging.info("=== 单张增强演示 ===")

    if not os.path.exists(SINGLE_IMAGE_PATH):
        logging.error(f"未找到示例图片 '{SINGLE_IMAGE_PATH}'")
        sys.exit(1)

    image = cv2.imread(SINGLE_IMAGE_PATH)
    if image is None:
        logging.error(f"无法加载图片 '{SINGLE_IMAGE_PATH}'")
        sys.exit(1)

    pipeline = load_pipeline_from_config(CONFIG_FILE)
    augmented_images, titles = apply_all_operations(image, pipeline)

    plot_all_grid([image], [augmented_images], [titles], SINGLE_SAVE_PATH)

def run_batch_demo():
    logging.info("=== 批量增强演示 ===")

    if not os.path.exists(BATCH_FOLDER):
        logging.error(f"未找到批量测试文件夹 '{BATCH_FOLDER}'")
        sys.exit(1)

    pipeline = load_pipeline_from_config(CONFIG_FILE)

    # 排除已生成的 augmented_comparison_grid 文件
    image_files = [
        f for f in os.listdir(BATCH_FOLDER)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        and "augmented_comparison_grid" not in f
    ]

    if not image_files:
        logging.warning(f"文件夹 '{BATCH_FOLDER}' 中未找到支持的图片文件")
        return

    images = []
    augmented_images_list = []
    titles_list = []

    for idx, fname in enumerate(image_files, 1):
        img_path = os.path.join(BATCH_FOLDER, fname)
        image = cv2.imread(img_path)
        if image is None:
            logging.warning(f"无法加载图片: {fname}")
            continue

        augmented_images, titles = apply_all_operations(image, pipeline)

        images.append(image)
        augmented_images_list.append(augmented_images)
        titles_list.append(titles)

        logging.info(f"[{idx}/{len(image_files)}] 已处理: {fname}")

    if images:
        plot_all_grid(images, augmented_images_list, titles_list, BATCH_SAVE_PATH)
    else:
        logging.warning("无有效图片进行批量增强")

if __name__ == "__main__":
    """
    使用：
    python augmentor_demo.py single   # 单张增强可视化
    python augmentor_demo.py batch    # 批量增强可视化
    """
    if len(sys.argv) < 2:
        logging.error("请指定运行模式: single 或 batch")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode == "single":
        run_single_demo()
    elif mode == "batch":
        run_batch_demo()
    else:
        logging.error("无效模式，请使用: single 或 batch")
        sys.exit(1)
