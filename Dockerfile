# Copyright (c) 2026 Aanand Tiwari
# BSD 3-Clause License

# Multi-stage build using openenv-base
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

ARG BUILD_MODE=standalone

# Copy environment code
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv > /dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install git (needed if pulling from git repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# --- Runtime stage ---
FROM ${BASE_IMAGE}

WORKDIR /app/env

# Copy virtual environment and code from builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Activate the venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the FastAPI server
CMD ["sh", "-c", "uvicorn math_reasoning_env.server.app:app --host 0.0.0.0 --port 8000"]
