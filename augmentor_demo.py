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

# === 日志设置 ===
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

def load_pipeline(config_file):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        pipeline = config.get('pipeline', [])
        if not pipeline:
            raise ValueError(f"{config_file} 中 pipeline 为空或不存在。")
        return pipeline
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        sys.exit(1)

def augment_single_image(image, pipeline, augmentor):
    augmented_list = []
    titles = []
    for operation in pipeline:
        op_name = operation['name']
        params = operation.get('params', {})
        try:
            method = getattr(augmentor, f"_{op_name}")
            augmented_image = method(image.copy(), **params)
            augmented_list.append(augmented_image)
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            titles.append(f"{op_name}({param_str})" if param_str else op_name)
        except Exception as e:
            logging.warning(f"增强 {op_name} 时出错，已跳过: {e}")
    return augmented_list, titles

def plot_comparisons(image_list, augmented_lists, titles_list, save_path):
    num_images = len(image_list)
    num_cols = len(augmented_lists[0]) + 1

    plt.figure(figsize=(5 * num_cols, 5 * num_images))

    for row_idx, (image, aug_list, titles) in enumerate(zip(image_list, augmented_lists, titles_list)):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        aug_rgbs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in aug_list]

        plt.subplot(num_images, num_cols, row_idx * num_cols + 1)
        plt.title("Original")
        plt.axis('off')
        plt.imshow(image_rgb)

        for col_idx, (aug_img, title) in enumerate(zip(aug_rgbs, titles)):
            plt.subplot(num_images, num_cols, row_idx * num_cols + col_idx + 2)
            plt.title(title)
            plt.axis('off')
            plt.imshow(aug_img)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    logging.info(f"已保存增强可视化结果到 {save_path}")

def run_single_demo():
    logging.info("=== 单张图片增强测试 ===")

    if not os.path.exists(SINGLE_IMAGE_PATH):
        logging.error(f"未找到示例图片 '{SINGLE_IMAGE_PATH}'，请准备好后再运行。")
        sys.exit(1)

    image = cv2.imread(SINGLE_IMAGE_PATH)
    if image is None:
        logging.error(f"无法加载图片 '{SINGLE_IMAGE_PATH}'，请检查路径和完整性。")
        sys.exit(1)

    pipeline = load_pipeline(CONFIG_FILE)
    augmentor = ImageAugmentor(config_path=CONFIG_FILE)

    augmented_list, titles = augment_single_image(image, pipeline, augmentor)
    plot_comparisons([image], [augmented_list], [titles], SINGLE_SAVE_PATH)

def run_batch_demo():
    logging.info("=== 批量图片增强测试 ===")

    if not os.path.exists(BATCH_FOLDER):
        logging.error(f"未找到批量测试文件夹 '{BATCH_FOLDER}'，请准备好后再运行。")
        sys.exit(1)

    pipeline = load_pipeline(CONFIG_FILE)
    augmentor = ImageAugmentor(config_path=CONFIG_FILE)

    image_list = []
    augmented_lists = []
    titles_list = []

    generated_filename = os.path.basename(BATCH_SAVE_PATH)
    image_files = [
        f for f in os.listdir(BATCH_FOLDER)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT and f != generated_filename
    ]

    if not image_files:
        logging.warning(f"文件夹 '{BATCH_FOLDER}' 中未找到支持的图片文件。")
        return

    for idx, fname in enumerate(image_files, 1):
        img_path = os.path.join(BATCH_FOLDER, fname)
        image = cv2.imread(img_path)
        if image is None:
            logging.warning(f"跳过无法加载的图片: {fname}")
            continue

        augmented_list, titles = augment_single_image(image, pipeline, augmentor)

        image_list.append(image)
        augmented_lists.append(augmented_list)
        titles_list.append(titles)

        logging.info(f"[{idx}/{len(image_files)}] 已处理: {fname}")

    if image_list:
        plot_comparisons(image_list, augmented_lists, titles_list, BATCH_SAVE_PATH)
    else:
        logging.warning("没有可用于批量可视化的有效图片，已跳过生成。")

if __name__ == "__main__":
    """
    使用：
    python augmentor_demo.py single   # 单张测试
    python augmentor_demo.py batch    # 批量测试
    """
    try:
        if len(sys.argv) < 2:
            logging.error("请指定运行模式参数: single 或 batch")
            sys.exit(1)

        mode = sys.argv[1].lower()
        if mode == "single":
            run_single_demo()
        elif mode == "batch":
            run_batch_demo()
        else:
            logging.error("无效模式，请使用: single 或 batch")
            sys.exit(1)
    except Exception as e:
        logging.error(f"运行时出错: {e}")
        sys.exit(1)
