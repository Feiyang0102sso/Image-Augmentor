import streamlit as st
import cv2
import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
from io import StringIO, BytesIO
import logging

# Import your provided modules
# Ensure augmentor_module.py and related dependencies are in the same directory
from augmentor_module import ImageAugmentor

# --- UNIFIED LOGGING SETUP ---
# 1. Create a custom handler that appends logs to a list
class ListLogHandler(logging.Handler):
    """A custom logging handler that appends formatted logs to a given list."""

    def __init__(self, log_list):
        super().__init__()
        self.log_list = log_list

    def emit(self, record):
        # self.format(record) creates the formatted log string
        msg = self.format(record)
        self.log_list.append(msg)


# 2. A function to set up the logger.
def setup_logging(log_list):
    """Configures the root logger to send messages to our custom handler."""
    root_logger = logging.getLogger()

    # Set the lowest level to capture all messages (INFO, WARNING, etc.)
    root_logger.setLevel(logging.INFO)

    # Create a formatter to match the console's style
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

    # Create our custom handler and set its formatter
    list_handler = ListLogHandler(log_list)
    list_handler.setFormatter(formatter)

    # Add handler to the root logger, but only if not already there
    if not any(isinstance(h, ListLogHandler) for h in root_logger.handlers):
        root_logger.addHandler(list_handler)


# --- Core Functional Logic ---
@st.cache_resource
def load_model():
    """Loads the pre-trained ResNet50 model."""
    loaded_model = models.resnet50(pretrained=True)
    loaded_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return loaded_model, device


@st.cache_data
def load_imagenet_classes():
    """Loads ImageNet class labels from a file."""
    try:
        with open('imagenet_classes.txt') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error("Error: `imagenet_classes.txt` not found.")
        return None


# --- Streamlit UI and App Logic ---
st.set_page_config(page_title="Image Augmentation & Inference", layout="wide")
st.title("🖼️ Image Augmentation & Inference Analyzer")

# Load model and classes
with st.spinner("Initializing model..."):
    model, device = load_model()
    classes = load_imagenet_classes()

if model and classes:
    st.toast("Model loaded successfully!", icon="🤖")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- Helper Functions ---
def load_augmentor_from_upload(uploaded_config_file):
    try:
        stringio = StringIO(uploaded_config_file.getvalue().decode("utf-8"))
        config_json = json.load(stringio)
        with open("temp_config.json", "w") as f:
            json.dump(config_json, f)
        augmentor_instance = ImageAugmentor("temp_config.json")
        os.remove("temp_config.json")
        return augmentor_instance
    except Exception as ex:
        st.error(f"Failed to load config file: {ex}")
        return None


def get_pipeline_info(augmentor_instance):
    titles = ["Original"]
    for op in augmentor_instance.pipeline:
        name = op["name"]
        params = op.get("params", {})
        if name == "RandomChoice":
            choices = [c["name"] for c in params.get("choices", [])]
            titles.append(f"RandomChoice({', '.join(choices)})")
        else:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            titles.append(f"{name}({param_str})" if param_str else name)
    return titles


def infer_image(image_cv):
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, predicted_idx = torch.max(probabilities, 0)
    return classes[predicted_idx], confidence.item()


def process_single_image(uploaded_file, augmentor_instance):
    """
    Processes a single image and uses the standard logging module for all output.
    It no longer returns a list of logs.
    """
    try:
        pil_img = Image.open(uploaded_file).convert("RGB")
        image_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        orig_pred, orig_conf = infer_image(image_cv)
        logging.info(f"File: {uploaded_file.name} | Original prediction: {orig_pred} ({orig_conf:.2%})")

        augmented_images, image_predictions = [], [(orig_pred, orig_conf)]
        current_image = image_cv.copy()

        for transform in augmentor_instance.transforms:
            current_image = transform(current_image)
            aug_pred, aug_conf = infer_image(current_image)
            augmented_images.append(current_image.copy())
            image_predictions.append((aug_pred, aug_conf))
            logging.info(f"Augmented ({transform.__class__.__name__}) prediction: {aug_pred} ({aug_conf:.2%})")

        return image_cv, augmented_images, image_predictions

    except Exception as ex:
        logging.error(f"An error occurred while processing {uploaded_file.name}: {ex}", exc_info=True)
        return None, None, None


def plot_and_display_results(original_images, all_aug_img_lists, all_preds, pipeline_titles):
    num_images = len(original_images)
    num_cols = len(all_aug_img_lists[0]) + 1
    fig = plt.figure(figsize=(5 * num_cols, 6 * num_images))
    for row_idx, (orig_img, aug_imgs) in enumerate(zip(original_images, all_aug_img_lists)):
        all_imgs = [orig_img] + aug_imgs
        all_img_preds = all_preds[row_idx]
        for col_idx, (img, pred) in enumerate(zip(all_imgs, all_img_preds)):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax = plt.subplot(num_images, num_cols, row_idx * num_cols + col_idx + 1)
            pred_text = f"{pred[0]}\n({pred[1]:.2%})"
            ax.set_title(f"{pipeline_titles[col_idx]}\n{pred_text}", fontsize=10)
            ax.axis('off')
            ax.imshow(img_rgb)
    plt.tight_layout(pad=2.0)
    st.pyplot(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    st.download_button("Download Results Image", buf.getvalue(), "augmentation_comparison.png", "image/png")
    plt.close(fig)


# --- Sidebar Controls ---
st.sidebar.header("⚙️ Control Panel")
mode = st.sidebar.radio("Select Mode", ('Single Image', 'Batch of Images'))
uploaded_files = st.sidebar.file_uploader("Upload Image(s)",
                                          type=['png', 'jpg', 'jpeg'],
                                          accept_multiple_files=(mode == 'Batch of Images'))
uploaded_config = st.sidebar.file_uploader("Upload config.json", type=['json'])

with st.sidebar.expander("See default config.json example"):
    st.code("""{...}""", language='json')  # Omitted for brevity

# --- Main Application Logic ---
if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=(not model or not classes)):
    if not uploaded_files or not uploaded_config:
        st.warning("Please upload image(s) and a config file.")
    else:
        # This list will hold all log messages for this run.
        session_logs = []
        setup_logging(session_logs)

        files_to_process = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]

        with st.spinner("Processing... please wait."):
            augmentor = load_augmentor_from_upload(uploaded_config)
            if augmentor:
                pipeline_titles = get_pipeline_info(augmentor)
                all_originals, all_aug_lists, all_preds = [], [], []

                for file in files_to_process:
                    original, aug_list, pred_list = process_single_image(file, augmentor)
                    if original is not None:
                        all_originals.append(original)
                        all_aug_lists.append(aug_list)
                        all_preds.append(pred_list)

                if all_originals:
                    st.header("📊 Visual Comparison of Results")
                    plot_and_display_results(all_originals, all_aug_lists, all_preds, pipeline_titles)

                    st.header("📝 Processing Logs")
                    # Display all captured logs in a scrollable code block
                    st.code("\n".join(session_logs), language="log")