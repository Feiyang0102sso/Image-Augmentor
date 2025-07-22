import cv2
import os
import sys
import matplotlib.pyplot as plt
import logging
import numpy as np
import argparse
import torch
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from augmentor_module import ImageAugmentor

# augmentor_demo.py

# === Configuration ===
SINGLE_IMAGE_PATH = "../demo_picture/single_demo.JPEG"
BATCH_FOLDER = "../demo_picture/batchs"
CONFIG_FILE = "config.json"
BATCH_SAVE_PATH = "../demo_picture/batchs/augmented_comparison_grid.png"
SINGLE_SAVE_PATH = "../demo_picture/single_augmented_comparison.png"
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

# Parse command line arguments
parser = argparse.ArgumentParser(description="Image augmentation demo with custom config file.")
parser.add_argument('mode', choices=['single', 'batch'], help='Mode of operation: single or batch')
parser.add_argument('--config', type=str, default=CONFIG_FILE, help='Path to the JSON config file')
args = parser.parse_args()
CONFIG_FILE = args.config
logging.info(f"Using config file: {CONFIG_FILE}")

# Initialize pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ImageNet class labels (simplified, should load full labels in practice)
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# Define preprocessing transforms for ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_augmentor(config_file):
    """Initialize the ImageAugmentor with the given config file."""
    try:
        config_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_file)
        logging.info(f"Attempting to load config file: {config_file_path}")
        augmentor = ImageAugmentor(config_file_path)
        logging.info(f"Successfully loaded config file: {config_file}")
        return augmentor
    except Exception as e:
        logging.error(f"Failed to initialize ImageAugmentor: {e}")
        raise e


def get_pipeline_info(augmentor):
    """Get titles for augmentation pipeline."""
    titles = []
    for op in augmentor.pipeline:
        name = op["name"]
        params = op.get("params", {})
        if name == "RandomChoice":
            choices = params.get("choices", [])
            choice_names = [c["name"] for c in choices]
            title = f"RandomChoice({', '.join(choice_names)})"
            titles.append(title)
        else:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            titles.append(f"{name}({param_str})" if param_str else name)
    return ["Original"] + titles


def infer_image(image, return_probs=False):
    """Run inference on a single image."""
    # Convert OpenCV image (BGR) to PIL image (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Apply preprocessing
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top prediction
    confidence, predicted_idx = torch.max(probabilities, 0)
    predicted_class = classes[predicted_idx]

    if return_probs:
        return predicted_class, confidence.item(), probabilities
    return predicted_class, confidence.item()


def plot_all_grid(images, augmented_images_list, titles, save_path, predictions=None):
    """Plot original and augmented images with predictions in a grid."""
    num_images = len(images)
    num_cols = len(augmented_images_list[0]) + 1

    plt.figure(figsize=(5 * num_cols, 6 * num_images))

    for row_idx, (orig_img, aug_imgs) in enumerate(zip(images, augmented_images_list)):
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # Plot original image
        plt.subplot(num_images, num_cols, row_idx * num_cols + 1)
        pred_text = f"{predictions[row_idx][0][0]} ({predictions[row_idx][0][1]:.2%})"
        plt.title(f"{titles[0]}\n{pred_text}")
        plt.axis('off')
        plt.imshow(orig_rgb)

        # Plot augmented images
        for col_idx, aug_img in enumerate(aug_imgs):
            aug_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
            plt.subplot(num_images, num_cols, row_idx * num_cols + col_idx + 2)
            pred_text = f"{predictions[row_idx][col_idx + 1][0]} ({predictions[row_idx][col_idx + 1][1]:.2%})"
            plt.title(f"{titles[col_idx + 1]}\n{pred_text}")
            plt.axis('off')
            plt.imshow(aug_rgb)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    logging.info(f"Saved visualization to {save_path}")


def generate_report(images, augmented_images_list, titles, predictions, SINGLE_SAVE_PATH):
    """Generate a visualization report with predictions."""
    plot_all_grid(images, augmented_images_list, titles, SINGLE_SAVE_PATH, predictions)

    # Generate markdown report
    with open('inference_report.md', 'w') as f:
        f.write("# Image Augmentation Inference Report\n\n")
        f.write(f"![Visualization]({SINGLE_SAVE_PATH})\n\n")
        f.write("## Prediction Results\n\n")

        for row_idx, (orig_img, aug_imgs) in enumerate(zip(images, augmented_images_list)):
            f.write(f"### Image {row_idx + 1}\n")
            f.write(f"- **Original**: {predictions[row_idx][0][0]} ({predictions[row_idx][0][1]:.2%})\n")
            for col_idx, aug_img in enumerate(aug_imgs):
                f.write(
                    f"- **{titles[col_idx + 1]}**: {predictions[row_idx][col_idx + 1][0]} ({predictions[row_idx][col_idx + 1][1]:.2%})\n")
            f.write("\n")


def run_single_demo():
    """Run demo for a single image with inference."""
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

    # Get original prediction
    orig_pred, orig_conf = infer_image(image)
    logging.info(f"Original image prediction: {orig_pred} ({orig_conf:.2%})")

    # Apply augmentations and get predictions
    augmented_images = []
    predictions = [[(orig_pred, orig_conf)]]
    current_image = image.copy()
    for transform in augmentor.transforms:
        current_image = transform(current_image)
        aug_pred, aug_conf = infer_image(current_image)
        augmented_images.append(current_image.copy())
        predictions[0].append((aug_pred, aug_conf))
        logging.info(f"Augmented image ({transform.__class__.__name__}) prediction: {aug_pred} ({aug_conf:.2%})")

    # Generate visualization and report
    generate_report([image], [augmented_images], titles, predictions, SINGLE_SAVE_PATH)


def run_batch_demo():
    """Run demo for a batch of images with inference."""
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
    predictions = []

    for idx, fname in enumerate(image_files, 1):
        img_path = os.path.join(BATCH_FOLDER, fname)
        image = cv2.imread(img_path)
        if image is None:
            logging.warning(f"Failed to load image: {fname}")
            continue

        try:
            # Get original prediction
            orig_pred, orig_conf = infer_image(image)
            logging.info(f"[{idx}/{len(image_files)}] Original {fname}: {orig_pred} ({orig_conf:.2%})")

            # Apply augmentations and get predictions
            augmented_images = []
            image_predictions = [(orig_pred, orig_conf)]
            current_image = image.copy()
            for transform in augmentor.transforms:
                current_image = transform(current_image)
                aug_pred, aug_conf = infer_image(current_image)
                augmented_images.append(current_image.copy())
                image_predictions.append((aug_pred, aug_conf))
                logging.info(
                    f"[{idx}/{len(image_files)}] Augmented {fname} ({transform.__class__.__name__}): {aug_pred} ({aug_conf:.2%})")

            images.append(image)
            augmented_images_list.append(augmented_images)
            predictions.append(image_predictions)
        except Exception as e:
            logging.warning(f"Augmentation failed for {fname}: {e}")

    if images:
        generate_report(images, augmented_images_list, titles, predictions, BATCH_SAVE_PATH)


if __name__ == "__main__":
    logging.info(f"Running in mode: {args.mode}")
    if args.mode == "single":
        run_single_demo()
    elif args.mode == "batch":
        run_batch_demo()
    else:
        logging.error("Invalid mode, use: single or batch")
        sys.exit(1)