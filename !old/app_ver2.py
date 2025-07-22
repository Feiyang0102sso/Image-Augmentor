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

# Import your provided modules
# Ensure augmentor_module.py and related dependencies are in the same directory
from augmentor_module import ImageAugmentor


# --- Core Functional Logic (No UI elements here) ---

@st.cache_resource
def load_model():
    """
    Loads the pre-trained ResNet50 model.
    This function is cached and MUST NOT contain any Streamlit UI calls.
    """
    loaded_model = models.resnet50(pretrained=True)
    loaded_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = loaded_model.to(device)
    return loaded_model, device


@st.cache_data
def load_imagenet_classes():
    """Loads ImageNet class labels from a file."""
    try:
        with open('imagenet_classes.txt') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except FileNotFoundError:
        st.error("Error: `imagenet_classes.txt` not found. Please ensure it is in the same directory as app.py.")
        return None


# --- Streamlit UI and App Logic ---

st.set_page_config(page_title="Image Augmentation & Inference", layout="wide")
st.title("🖼️ Image Augmentation & Inference Analyzer")
st.markdown(
    "Upload images and a configuration file to run an augmentation pipeline and see its effect on model predictions.")

# Load model and classes with user feedback
with st.spinner("Initializing model... This may take a moment on first run."):
    model, device = load_model()
    classes = load_imagenet_classes()

if model and classes:
    st.toast("Model loaded successfully!", icon="🤖")

# Define the preprocessing transforms for ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- Helper Functions ---

def load_augmentor_from_upload(uploaded_config_file):
    """Initializes ImageAugmentor from an uploaded config file."""
    try:
        stringio = StringIO(uploaded_config_file.getvalue().decode("utf-8"))
        config_json = json.load(stringio)

        temp_config_path = "temp_config.json"
        with open(temp_config_path, "w") as f:
            json.dump(config_json, f)

        augmentor_instance = ImageAugmentor(temp_config_path)
        os.remove(temp_config_path)
        return augmentor_instance
    except Exception as ex:
        st.error(f"Failed to load or parse config file: {ex}")
        return None


def get_pipeline_info(augmentor_instance):
    """Gets titles for the augmentation pipeline steps."""
    pipeline_titles = []
    for op in augmentor_instance.pipeline:
        name = op["name"]
        params = op.get("params", {})
        if name == "RandomChoice":
            choices = params.get("choices", [])
            choice_names = [c["name"] for c in choices]
            title = f"RandomChoice({', '.join(choice_names)})"
            pipeline_titles.append(title)
        else:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            pipeline_titles.append(f"{name}({param_str})" if param_str else name)
    return ["Original"] + pipeline_titles


def infer_image(image_cv):
    """Runs inference on a single OpenCV image."""
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
    """Reads, augments, and runs inference, returning results and logs."""
    logs = []
    try:
        pil_img = Image.open(uploaded_file).convert("RGB")
        image_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        orig_pred, orig_conf = infer_image(image_cv)
        logs.append(f"INFO: File: {uploaded_file.name} | Original prediction: {orig_pred} ({orig_conf:.2%})")

        augmented_images, image_predictions = [], [(orig_pred, orig_conf)]
        current_image = image_cv.copy()

        for transform in augmentor_instance.transforms:
            current_image = transform(current_image)
            aug_pred, aug_conf = infer_image(current_image)
            augmented_images.append(current_image.copy())
            image_predictions.append((aug_pred, aug_conf))
            logs.append(f"INFO: ...Augmented ({transform.__class__.__name__}) prediction: {aug_pred} ({aug_conf:.2%})")

        return image_cv, augmented_images, image_predictions, logs

    except Exception as ex:
        error_msg = f"ERROR: An error occurred while processing {uploaded_file.name}: {ex}"
        st.error(error_msg)
        logs.append(error_msg)
        return None, None, None, logs


def plot_and_display_results(original_images, all_aug_img_lists, all_preds, pipeline_titles):
    """Plots and displays the results grid in Streamlit."""
    num_images = len(original_images)
    num_cols = len(all_aug_img_lists[0]) + 1

    fig = plt.figure(figsize=(5 * num_cols, 6 * num_images))

    for row_idx, (orig_img, aug_imgs) in enumerate(zip(original_images, all_aug_img_lists)):
        # Plot original and augmented images with their predictions
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
uploaded_files = st.sidebar.file_uploader(
    "Upload Image(s)",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'],
    accept_multiple_files=(mode == 'Batch of Images')
)
uploaded_config = st.sidebar.file_uploader("Upload config.json", type=['json'])

with st.sidebar.expander("See default config.json example"):
    st.code("""{
  "pipeline": [
    { "name": "GaussianBlur", "params": { "kernel_size": 5 } },
    { "name": "Rotate", "params": { "angle": 15 } },
    { "name": "RandomChoice",
      "params": { "choices": [
        { "name": "Flip", "params": {"flip_code": 1} },
        { "name": "SaltAndPepperNoise", "params": {"amount": 0.05} }
      ]}
    }
  ]
}""", language='json')

# --- Main Application Logic ---
if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=(not model or not classes)):
    if not uploaded_files:
        st.warning("Please upload at least one image.")
    elif not uploaded_config:
        st.warning("Please upload a `config.json` file.")
    else:
        # Re-pack single file into a list for consistent processing
        files_to_process = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]

        with st.spinner("Processing... please wait."):
            augmentor = load_augmentor_from_upload(uploaded_config)
            if augmentor:
                pipeline_titles = get_pipeline_info(augmentor)
                all_originals, all_aug_lists, all_preds, all_logs = [], [], [], []

                log_expander = st.expander("View Processing Logs")
                progress_bar = st.progress(0, text="Processing progress")

                for i, file in enumerate(files_to_process):
                    original, aug_list, pred_list, logs = process_single_image(file, augmentor)
                    if original is not None:
                        all_originals.append(original)
                        all_aug_lists.append(aug_list)
                        all_preds.append(pred_list)
                    all_logs.extend(logs)

                    with log_expander:
                        st.text("\n".join(all_logs))
                    progress_bar.progress((i + 1) / len(files_to_process), text=f"Processing: {file.name}")

                progress_bar.empty()

                if all_originals:
                    st.header("📊 Visual Comparison of Results")
                    plot_and_display_results(all_originals, all_aug_lists, all_preds, pipeline_titles)

else:
    if model and classes:
        st.info("Please configure settings in the sidebar and click **Run Analysis**.")
    else:
        st.error(
            "Model or class labels failed to load. Please check the console for errors and ensure `imagenet_classes.txt` exists.")