import cv2
import os
import sys
import matplotlib.pyplot as plt
import logging
import numpy as np
from augmentor_module import ImageAugmentor

# === Configuration ===
SINGLE_IMAGE_PATH = "demo_picture/single_demo.JPEG"
BATCH_FOLDER = "demo_picture/batchs"
CONFIG_FILE = "config.json"
# config.json config_random_choice.json
SINGLE_SAVE_PATH = "demo_picture/single_augmented_comparison.png"
BATCH_SAVE_PATH = "demo_picture/batchs/augmented_comparison_grid.png"
SUPPORTED_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("demo_log.txt"),
        logging.StreamHandler()
    ]
)


def load_augmentor(config_file):
    """Initialize the ImageAugmentor with the given config file."""
    try:
        augmentor = ImageAugmentor(config_file)
        return augmentor
    except Exception as e:
        logging.error(f"Failed to initialize ImageAugmentor: {e}")
        raise e
        # sys.exit(1)

def get_pipeline_info(augmentor):
    titles = []
    for op in augmentor.pipeline:
        name = op["name"]
        params = op.get("params", {})

        if name == "RandomChoice":
            # 提取 choices 列表中每个子增强的 name
            choices = params.get("choices", [])
            choice_names = [c["name"] for c in choices]
            title = f"RandomChoice({', '.join(choice_names)})"
            titles.append(title)
        else:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            titles.append(f"{name}({param_str})" if param_str else name)
    return ["Original"] + titles


def plot_all_grid(images, augmented_images_list, titles, save_path):
    """Plot original and augmented images in a grid."""
    num_images = len(images)
    num_cols = len(augmented_images_list[0]) + 1  # Original + augmented

    plt.figure(figsize=(5 * num_cols, 5 * num_images))

    for row_idx, (orig_img, aug_imgs) in enumerate(zip(images, augmented_images_list)):
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        plt.subplot(num_images, num_cols, row_idx * num_cols + 1)
        plt.title(titles[0])  # Use "Original" as the title for the first column
        plt.axis('off')
        plt.imshow(orig_rgb)

        for col_idx, aug_img in enumerate(aug_imgs):
            aug_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
            plt.subplot(num_images, num_cols, row_idx * num_cols + col_idx + 2)
            plt.title(titles[col_idx + 1])  # Use subsequent titles for augmented images
            plt.axis('off')
            plt.imshow(aug_rgb)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    logging.info(f"Saved visualization to {save_path}")

def run_single_demo():
    """Run demo for a single image."""
    logging.info("=== Single Image Augmentation Demo ===")

    if not os.path.exists(SINGLE_IMAGE_PATH):
        logging.error(f"Image not found: {SINGLE_IMAGE_PATH}")
        sys.exit(1)

    image = cv2.imread(SINGLE_IMAGE_PATH)
    if image is None:
        logging.error(f"Failed to load image: {SINGLE_IMAGE_PATH}")
        sys.exit(1)

    augmentor = load_augmentor(CONFIG_FILE)
    titles = get_pipeline_info(augmentor)

    # Apply each transformation individually
    augmented_images = []
    current_image = image.copy()
    for transform in augmentor.transforms:
        current_image = transform(current_image)
        augmented_images.append(current_image.copy())

    plot_all_grid([image], [augmented_images], titles, SINGLE_SAVE_PATH)

def run_batch_demo():
    """Run demo for a batch of images."""
    logging.info("=== Batch Image Augmentation Demo ===")

    if not os.path.exists(BATCH_FOLDER):
        logging.error(f"Batch folder not found: {BATCH_FOLDER}")
        sys.exit(1)

    image_files = [
        f for f in os.listdir(BATCH_FOLDER)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        and "augmented_comparison_grid" not in f
    ]

    if not image_files:
        logging.warning(f"No supported images found in {BATCH_FOLDER}")
        return

    augmentor = load_augmentor(CONFIG_FILE)
    titles = get_pipeline_info(augmentor)
    images = []
    augmented_images_list = []

    for idx, fname in enumerate(image_files, 1):
        img_path = os.path.join(BATCH_FOLDER, fname)
        image = cv2.imread(img_path)
        if image is None:
            logging.warning(f"Failed to load image: {fname}")
            continue

        try:
            # Apply each transformation individually
            augmented_images = []
            current_image = image.copy()
            for transform in augmentor.transforms:
                current_image = transform(current_image)
                augmented_images.append(current_image.copy())
            images.append(image)
            augmented_images_list.append(augmented_images)
            logging.info(f"[{idx}/{len(image_files)}] Processed: {fname}")
        except Exception as e:
            logging.warning(f"Augmentation failed for {fname}: {e}")

    if images:
        plot_all_grid(images, augmented_images_list, titles, BATCH_SAVE_PATH)
    else:
        logging.warning("No valid images for batch augmentation")

if __name__ == "__main__":
    """
    Usage:
    python augmentor_demo.py single   # Single image augmentation visualization
    python augmentor_demo.py batch    # Batch image augmentation visualization
    """
    if len(sys.argv) < 2:
        logging.error("Please specify mode: single or batch")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode == "single":
        run_single_demo()
    elif mode == "batch":
        run_batch_demo()
    else:
        logging.error("Invalid mode, use: single or batch")
        sys.exit(1)