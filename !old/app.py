import streamlit as st
import cv2
import os
import logging
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
from io import StringIO, BytesIO

# 导入您提供的模块
# 确保 augmentor_module.py 和相关依赖项在同一目录下
from augmentor_module import ImageAugmentor

# --- 基本配置 ---

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    # stream=st.log  # 将日志输出到 Streamlit 界面
)


# --- 模型和数据加载 (带缓存) ---

@st.cache_resource
def load_model():
    """加载预训练的 ResNet50 模型"""
    model = models.resnet50(pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


@st.cache_data
def load_imagenet_classes():
    """加载 ImageNet 类别标签"""
    # 假设 imagenet_classes.txt 在同一目录下
    try:
        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    except FileNotFoundError:
        st.error("错误: `imagenet_classes.txt` 文件未找到。请确保它与 app.py 在同一目录下。")
        return None


# 初始化模型和类别
model, device = load_model()
classes = load_imagenet_classes()

# 定义 ResNet 的预处理转换
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- 核心功能函数 (从 augmentor_demo.py 改编) ---

def load_augmentor_from_upload(uploaded_config):
    """从上传的配置文件初始化 ImageAugmentor"""
    try:
        # 将上传的文件内容解码为字符串，然后加载为 JSON
        stringio = StringIO(uploaded_config.getvalue().decode("utf-8"))
        config_str = stringio.read()
        config_json = json.loads(config_str)

        # ImageAugmentor 需要一个文件路径，所以我们临时创建一个
        with open("temp_config.json", "w") as f:
            json.dump(config_json, f)

        augmentor = ImageAugmentor("temp_config.json")
        logging.info("成功加载并解析上传的配置文件。")
        os.remove("temp_config.json")  # 清理临时文件
        return augmentor
    except Exception as e:
        st.error(f"加载或解析配置文件失败: {e}")
        logging.error(f"加载或解析配置文件失败: {e}")
        return None


def get_pipeline_info(augmentor):
    """获取增强流程的标题信息"""
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


def infer_image(image_cv):
    """对单个 OpenCV 图像进行推理"""
    if classes is None: return "标签文件缺失", 0.0

    # 将 OpenCV 图像 (BGR) 转换为 PIL 图像 (RGB)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # 应用预处理
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # 运行推理
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # 获取最高置信度的预测结果
    confidence, predicted_idx = torch.max(probabilities, 0)
    predicted_class = classes[predicted_idx]

    return predicted_class, confidence.item()


def plot_and_display_results(images, augmented_images_list, titles, predictions):
    """在 Streamlit 中绘制并显示结果网格"""
    num_images = len(images)
    if num_images == 0:
        st.warning("没有可供显示的图像。")
        return

    num_cols = len(augmented_images_list[0]) + 1

    fig = plt.figure(figsize=(5 * num_cols, 6 * num_images))

    for row_idx, (orig_img, aug_imgs) in enumerate(zip(images, augmented_images_list)):
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # 绘制原始图像
        ax = plt.subplot(num_images, num_cols, row_idx * num_cols + 1)
        pred_text = f"{predictions[row_idx][0][0]}\n({predictions[row_idx][0][1]:.2%})"
        ax.set_title(f"{titles[0]}\n{pred_text}", fontsize=10)
        ax.axis('off')
        ax.imshow(orig_rgb)

        # 绘制增强后的图像
        for col_idx, aug_img in enumerate(aug_imgs):
            aug_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
            ax = plt.subplot(num_images, num_cols, row_idx * num_cols + col_idx + 2)
            pred_text = f"{predictions[row_idx][col_idx + 1][0]}\n({predictions[row_idx][col_idx + 1][1]:.2%})"
            ax.set_title(f"{titles[col_idx + 1]}\n{pred_text}", fontsize=10)
            ax.axis('off')
            ax.imshow(aug_rgb)

    plt.tight_layout(pad=2.0)
    st.pyplot(fig)

    # 提供图像下载按钮
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    st.download_button(
        label="下载结果图",
        data=buf.getvalue(),
        file_name="augmentation_comparison.png",
        mime="image/png"
    )

    plt.close(fig)


def generate_report_markdown(images, titles, predictions):
    """生成 Markdown 格式的报告"""
    md_report = "## 推理报告详情\n\n"
    for row_idx, orig_img in enumerate(images):
        md_report += f"### 图像 {row_idx + 1}\n"
        md_report += f"- **{titles[0]}**: `{predictions[row_idx][0][0]}` (置信度: {predictions[row_idx][0][1]:.2%})\n"
        for col_idx in range(len(predictions[row_idx]) - 1):
            md_report += f"- **{titles[col_idx + 1]}**: `{predictions[row_idx][col_idx + 1][0]}` (置信度: {predictions[row_idx][col_idx + 1][1]:.2%})\n"
        md_report += "\n"
    return md_report


# --- Streamlit UI 界面 ---

st.set_page_config(page_title="图像增强与推理", layout="wide")
st.title("🖼️ 图像增强与推理分析器")
st.markdown("上传图像和配置文件，运行增强流程，并观察其对模型推理结果的影响。")

# --- 侧边栏 ---
st.sidebar.header("⚙️ 控制面板")

mode = st.sidebar.radio("选择模式", ('单张图像处理', '批量图像处理'))

uploaded_files = None
if mode == '单张图像处理':
    uploaded_files = st.sidebar.file_uploader(
        "上传一张图像",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'],
        accept_multiple_files=False
    )
    if uploaded_files:
        uploaded_files = [uploaded_files]  # 包装成列表以统一处理
else:
    uploaded_files = st.sidebar.file_uploader(
        "上传多张图像",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'],
        accept_multiple_files=True
    )

# 配置文件上传
st.sidebar.subheader("增强配置")
uploaded_config = st.sidebar.file_uploader("上传 config.json", type=['json'])

# 提供一个默认的config.json示例
DEFAULT_CONFIG = """
{
  "pipeline": [
    {
      "name": "GaussianBlur",
      "params": {
        "kernel_size": 5
      },
      "prob": 1.0
    },
    {
      "name": "Rotate",
      "params": {
        "angle": 15
      },
      "prob": 1.0
    },
    {
        "name": "RandomChoice",
        "params": {
            "choices": [
                {
                    "name": "Flip",
                    "params": {"flip_code": 1}
                },
                {
                    "name": "SaltAndPepperNoise",
                    "params": {"amount": 0.05}
                }
            ]
        }
    }
  ]
}
"""
with st.sidebar.expander("查看默认 config.json 示例"):
    st.code(DEFAULT_CONFIG, language='json')

# --- 主逻辑 ---
if st.sidebar.button("🚀 开始运行", type="primary", use_container_width=True):
    if not uploaded_files:
        st.warning("请先上传至少一张图像。")
    elif not uploaded_config:
        st.warning("请上传一个 `config.json` 配置文件。")
    elif classes is None:
        # 错误已在 load_imagenet_classes 中显示
        pass
    else:
        with st.spinner("正在处理中，请稍候..."):
            augmentor = load_augmentor_from_upload(uploaded_config)

            if augmentor:
                titles = get_pipeline_info(augmentor)

                # 用于存储所有结果的列表
                all_original_images = []
                all_augmented_images_list = []
                all_predictions = []

                progress_bar = st.progress(0, text="处理进度")

                for i, uploaded_file in enumerate(uploaded_files):
                    # 读取和转换图像
                    pil_image = Image.open(uploaded_file).convert("RGB")
                    image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                    try:
                        # 对原始图像进行推理
                        orig_pred, orig_conf = infer_image(image_cv)
                        logging.info(f"文件: {uploaded_file.name} | 原始图像预测: {orig_pred} ({orig_conf:.2%})")

                        # 应用增强流程并进行推理
                        augmented_images = []
                        image_predictions = [(orig_pred, orig_conf)]
                        current_image = image_cv.copy()

                        for transform in augmentor.transforms:
                            current_image = transform(current_image)
                            aug_pred, aug_conf = infer_image(current_image)
                            augmented_images.append(current_image.copy())
                            image_predictions.append((aug_pred, aug_conf))
                            logging.info(
                                f"...增强后 ({transform.__class__.__name__}) 预测: {aug_pred} ({aug_conf:.2%})")

                        # 保存结果
                        all_original_images.append(image_cv)
                        all_augmented_images_list.append(augmented_images)
                        all_predictions.append(image_predictions)

                    except Exception as e:
                        st.error(f"处理文件 {uploaded_file.name} 时发生错误: {e}")
                        logging.error(f"处理文件 {uploaded_file.name} 时发生错误: {e}")

                    # 更新进度条
                    progress_bar.progress((i + 1) / len(uploaded_files), text=f"正在处理: {uploaded_file.name}")

                progress_bar.empty()

                if all_original_images:
                    st.header("📊 可视化对比结果")
                    plot_and_display_results(all_original_images, all_augmented_images_list, titles, all_predictions)

                    st.header("📝 报告摘要")
                    md_report = generate_report_markdown(all_original_images, titles, all_predictions)
                    st.markdown(md_report)

else:
    st.info("请在左侧侧边栏中配置参数，然后点击 **开始运行**。")