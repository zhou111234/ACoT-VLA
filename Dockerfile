# ================================================================
# Stage 1: 工具注入
# ================================================================
FROM ghcr.io/astral-sh/uv:0.6.0 AS uv-binary

# ================================================================
# Stage 2: 安装/编译阶段 (Builder)
# ================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

# 云端构建不需要限制并发，利用高速带宽
ENV UV_HTTP_TIMEOUT=600

COPY --from=uv-binary /uv /uvx /usr/local/bin/

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    patchelf \
    build-essential \
    git \
    git-lfs \
    ffmpeg \
    ca-certificates \
    && git lfs install \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/ACoT-VLA

# 拷贝项目文件
COPY . .

# 1. 执行 uv 同步 
# 注意：确保你的 pyproject.toml 已经配好了 lerobot 和 dlimp 的 git 地址或子模块
RUN uv sync && uv pip install -e .

# 2. 覆盖安装 JAX CUDA12 GPU 版
RUN uv pip uninstall jax jaxlib && \
    uv pip install "jax[cuda12_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3. 应用 Transformers 补丁 (添加一个判断防止文件夹不存在报错)
RUN if [ -d "src/openpi/models_pytorch/transformers_replace" ]; then \
    cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/; \
    fi

# ================================================================
# Stage 3: 运行阶段 (Runtime)
# ================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
ENV PYTHONPATH="/app/ACoT-VLA/src"
ENV VIRTUAL_ENV=/app/ACoT-VLA/.venv
ENV PATH="/app/ACoT-VLA/.venv/bin:${PATH}"
ENV LD_LIBRARY_PATH="/app/ACoT-VLA/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    python3.11 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    libglfw3 \
    ffmpeg \
    ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/ACoT-VLA /app/ACoT-VLA

WORKDIR /app/ACoT-VLA

EXPOSE 8000
# 补全了末尾参数
CMD ["bash", "scripts/server.sh", "0", "8000"]
