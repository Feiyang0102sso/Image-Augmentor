
FROM python:3.10-slim
LABEL authors="Feiyang"

# 安装 OpenCV 所需的库
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

# 设置工作目录
WORKDIR /app

# 复制项目代码到容器内的工作目录
COPY . .

# 复制并安装项目的依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 公开端口（Streamlit 默认端口为 8501）
EXPOSE 8501

# 设置容器启动时的命令，运行 Streamlit 应用
CMD ["streamlit", "run", "app.py"]
