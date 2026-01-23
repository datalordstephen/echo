FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# Constants
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PORT=8000

# copy and install dependencies only to cache them
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,id=s/c217ba62-5273-481e-a1a2-df0576ec1696-/root/cache/uv,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# copy application code
COPY src/ ./src/
COPY model/ ./model/

# install project and ensure given lockfile is used (--frozen) 
RUN --mount=type=cache,id=s/c217ba62-5273-481e-a1a2-df0576ec1696-/root/cache/uv,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
