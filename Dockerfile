# Multi-stage build. The builder resolves the locked dependencies with uv (CPU-only torch on Linux), installs
# the package and trains the serving bundle, so the runtime image starts in about a second. The runtime stage
# copies only the virtual environment, the simulated world and the artifacts, and runs as a non-root user.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 OMP_NUM_THREADS=2 \
    FRAUDNET_DATA_DIR=/app/data FRAUDNET_ARTIFACT_DIR=/app/artifacts
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
RUN /app/.venv/bin/fraudnet train --seed 42

FROM python:3.13-slim-bookworm AS runtime
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 \
    FRAUDNET_DATA_DIR=/app/data FRAUDNET_ARTIFACT_DIR=/app/artifacts
RUN groupadd --system fraudnet && useradd --system --gid fraudnet --home /app --shell /usr/sbin/nologin fraudnet
WORKDIR /app
COPY --from=builder --chown=fraudnet:fraudnet /app/.venv /app/.venv
COPY --from=builder --chown=fraudnet:fraudnet /app/data /app/data
COPY --from=builder --chown=fraudnet:fraudnet /app/artifacts /app/artifacts
USER fraudnet
EXPOSE 8772
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8772/health', timeout=4).status == 200 else 1)"]
CMD ["fraudnet", "serve", "--host", "0.0.0.0", "--port", "8772"]
