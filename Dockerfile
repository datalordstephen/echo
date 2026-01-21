FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Constants
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# Railway supplies PORT env var
ENV PORT=8000

# Working Directory
WORKDIR /app

# Sync dependencies
# Copy lockfile and pyproject
COPY pyproject.toml uv.lock ./

# Install dependencies only
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY src/ ./src/
COPY model/ ./model/

# Install project
RUN uv sync --frozen --no-dev

# Path
ENV PATH="/app/.venv/bin:$PATH"

# We use shell form to allow variable expansion for $PORT
CMD uvicorn src.api:app --host 0.0.0.0 --port $PORT
