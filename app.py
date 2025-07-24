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

from augmentor_module import ImageAugmentor

# --- UNIFIED LOGGING SETUP ---
class ListLogHandler(logging.Handler):
    def __init__(self, log_list):
        super().__init__()
        self.log_list = log_list

    def emit(self, record):
        msg = self.format(record)
        self.log_list.append(msg)


def setup_logging(log_list):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

    if not any(isinstance(h, ListLogHandler) for h in root_logger.handlers):
        list_handler = ListLogHandler(log_list)
        list_handler.setFormatter(formatter)
        root_logger.addHandler(list_handler)

    for h in root_logger.handlers:
        if isinstance(h, ListLogHandler):
            h.log_list = log_list


# --- Core Functional Logic ---
@st.cache_resource
def load_model():
    loaded_model = models.resnet50(pretrained=True)
    loaded_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return loaded_model, device


@st.cache_data
def load_imagenet_classes():
    try:
        with open('CLI_demo/imagenet_classes.txt') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error("Error: `imagenet_classes.txt` not found.")
        return None


# --- Streamlit UI and App Logic ---
st.set_page_config(page_title="Image Augmentation & Inference", layout="wide")
st.title("Image Augmentation & Inference Analyzer")

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


# --- Helper Functions (Unchanged) ---
def load_augmentor_from_file_object(config_file_object):
    try:
        stringio = StringIO(config_file_object.getvalue().decode("utf-8"))
        config_json = json.load(stringio)
        with open("temp_config.json", "w") as f:
            json.dump(config_json, f)
        augmentor_instance = ImageAugmentor("temp_config.json")
        os.remove("temp_config.json")
        return augmentor_instance
    except Exception as ex:
        st.error(f"Failed to load or parse config file: {ex}")
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
    # The download button now uses data from the function arguments.
    st.download_button("Download Results Image", buf.getvalue(), "augmentation_comparison.png", "image/png")
    plt.close(fig)


# <<< --- MODIFIED SECTION START (Session State) --- >>>

# 1. Initialize session state variables at the beginning.
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.logs = []
    st.session_state.originals = []
    st.session_state.aug_lists = []
    st.session_state.preds = []
    st.session_state.titles = []


# 2. Create a function to reset the state. This is good practice.
def reset_analysis_state():
    st.session_state.analysis_complete = False
    st.session_state.logs = []
    st.session_state.originals = []
    st.session_state.aug_lists = []
    st.session_state.preds = []
    st.session_state.titles = []


# --- Sidebar Controls ---
st.sidebar.header("Control Panel")
mode = st.sidebar.radio("Select Mode", ('Single Image', 'Batch of Images'), on_change=reset_analysis_state)
uploaded_files = st.sidebar.file_uploader(
    "Upload Image(s)", type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=(mode == 'Batch of Images'),
    on_change=reset_analysis_state  # Reset if files change
)
uploaded_config = st.sidebar.file_uploader(
    "Upload config.json", type=['json'],
    on_change=reset_analysis_state  # Reset if config changes
)

if uploaded_config is not None:
    expander_title = f"Current Config: {uploaded_config.name}"
    config_to_display = uploaded_config.getvalue().decode("utf-8")
else:
    expander_title = "Default Config (config.json)"
    try:
        with open("config.json", "r") as f:
            config_to_display = f.read()
    except FileNotFoundError:
        config_to_display = "Default 'config.json' not found in project directory."

with st.sidebar.expander(expander_title):
    st.code(config_to_display, language='json')

# --- Main Application Logic ---
if st.sidebar.button("Run Analysis", type="primary", use_container_width=True, disabled=(not model or not classes)):
    if not uploaded_files:
        st.warning("Please upload at least one image.")
        st.stop()

    config_to_use = None
    if uploaded_config is not None:
        config_to_use = uploaded_config
    else:
        try:
            with open("config.json", "rb") as f:
                config_to_use = BytesIO(f.read())
        except FileNotFoundError:
            st.error("Operation failed: No config file uploaded and default 'config.json' was not found.")
            st.stop()

    # Reset state before running a new analysis
    reset_analysis_state()
    session_logs = []
    setup_logging(session_logs)

    files_to_process = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]

    with st.spinner("Processing... please wait."):
        augmentor = load_augmentor_from_file_object(config_to_use)
        if augmentor:
            pipeline_titles = get_pipeline_info(augmentor)
            all_originals, all_aug_lists, all_preds = [], [], []

            for file in files_to_process:
                original, aug_list, pred_list = process_single_image(file, augmentor)
                if original is not None:
                    all_originals.append(original)
                    all_aug_lists.append(aug_list)
                    all_preds.append(pred_list)

            # 3. If analysis was successful, SAVE the results to session state.
            if all_originals:
                st.session_state.analysis_complete = True
                st.session_state.logs = session_logs
                st.session_state.originals = all_originals
                st.session_state.aug_lists = all_aug_lists
                st.session_state.preds = all_preds
                st.session_state.titles = pipeline_titles

# 4. This block now handles DISPLAYING the results from session state.
# It runs every time, but only shows output if the analysis_complete flag is True.
if st.session_state.analysis_complete:
    st.header("Processing Logs")
    st.code("\n".join(st.session_state.logs), language="log")

    st.header("Visual Comparison of Results")
    plot_and_display_results(
        st.session_state.originals,
        st.session_state.aug_lists,
        st.session_state.preds,
        st.session_state.titles
    )
else:
    # Show an initial message if no analysis has been run yet.
    st.info("Please upload image(s), select a config, and click 'Run Analysis'.")

# <<< --- MODIFIED SECTION END --- >>>