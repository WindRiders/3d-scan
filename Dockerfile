# 3d-scan Docker 部署

# ── 构建 ──
# docker build -t 3d-scan:latest .

# ── 运行 ──
# docker compose up -d

# ── GPU 单次任务 ──
# docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output 3d-scan:latest python3 scripts/e2e_3dgs_test.py

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV OPENBLAS_NUM_THREADS=4
ENV HF_ENDPOINT=https://hf-mirror.com

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# pip 依赖（分层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GPU 依赖（镜像较大，按需安装）
# 构建 GPU 镜像: docker build --build-arg INSTALL_GPU=1 -t 3d-scan:gpu .
ARG INSTALL_GPU=0
RUN if [ "$INSTALL_GPU" = "1" ]; then \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir xformers && \
    pip install --no-cache-dir dust3r@git+https://github.com/naver/dust3r.git; \
    fi

# 源码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]