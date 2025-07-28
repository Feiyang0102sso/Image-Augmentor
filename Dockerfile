# =================================================================
# 第一阶段：构建器 (Builder)
# =================================================================
FROM python:3.10-slim AS builder

LABEL authors="Feiyang"
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1 torchvision==0.22.1


# 复制并安装 requirements.txt 中的其他依赖
# 确保 torch 和 torchvision 已从 requirements.txt 中移除
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


# =================================================================
# 第二阶段：最终镜像 (Final Image)
# =================================================================
FROM python:3.10-slim

LABEL authors="Feiyang"
WORKDIR /app

# 安装运行时的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 从构建器阶段复制已安装的 Python 包
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制项目代码
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]