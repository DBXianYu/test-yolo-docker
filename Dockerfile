# ============== 终极精简版本 - 无PyTorch ==============
FROM nvidia/cuda:13.0.0-base-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# 镜像源
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 最小依赖包 - 使用最新ONNX Runtime版本
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    onnxruntime \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6 \
    Pillow==10.1.0 \
    numpy==1.24.3 \
    opencv-python-headless==4.8.1.78

# 清理
RUN find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + || true

# ============== 运行时 ==============
FROM nvidia/cuda:13.0.0-base-ubuntu22.04

ENV PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
RUN useradd -r -u 1000 appuser && chown appuser:appuser /app

# 复制应用和模型文件
COPY --chown=appuser:appuser main.py main.py
COPY --chown=appuser:appuser defect_yolov8n.engine .
COPY --chown=appuser:appuser defect_yolov8n.onnx .
COPY --chown=appuser:appuser defect_yolov8n.pt .

USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
