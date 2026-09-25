# DocuOCR Enterprise — ARM64 + 2GB RAM 特化版
FROM python:3.10-slim-bookworm

ENV TZ=Asia/Hong_Kong \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # ---- 2 核線程管制：唔畀 numpy/OpenBLAS/OpenCV 同 onnxruntime 爭 CPU ----
    # onnxruntime 已經用 intra_op=2；BLAS 限 1 條線避免超額訂閱導致頻繁切換
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    # ---- 2GB 慳 RAM 三件套 ----
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

# 只裝 runtime 需要嘅系統庫。
# 唔裝 build-essential / cmake：預設唔編譯 llama-cpp，build 快好多。
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f >/dev/null 2>&1 || true

ARG INSTALL_SLM=false

COPY requirements.txt requirements-slm.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---- 可選：Qwen2.5-0.5B 語境潤飾 ----
# 本機建置：docker build --build-arg INSTALL_SLM=true .
# GitHub Actions 會自動產出 base / slm 兩個 ARM64 版本。
# 注意：CI 用 QEMU 模擬 ARM64 編譯，刻意唔設 GGML_NATIVE 以保跨機兼容；
# 如喺 NAS 本機編譯想榨乾效能，可自行加 CMAKE_ARGS="-DGGML_NATIVE=ON"。
RUN if [ "$INSTALL_SLM" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
        && rm -rf /var/lib/apt/lists/* \
        && pip install --no-cache-dir -r requirements-slm.txt; \
    fi

COPY core/ ./core/
COPY lexicons/ ./lexicons/
COPY static/ ./static/
COPY main.py .

EXPOSE 8000

# 用 python 內建 urllib 做 healthcheck，唔使額外裝 curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
