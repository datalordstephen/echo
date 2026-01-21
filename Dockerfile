FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies
RUN dnf install -y libsndfile

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Constants
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Sync dependencies
# Copy lockfile and pyproject
WORKDIR ${LAMBDA_TASK_ROOT}
COPY pyproject.toml uv.lock ${LAMBDA_TASK_ROOT}

# Install dependencies only (optimization)
# --frozen: fail if lockfile is out of sync
# --no-dev: production only
# --no-install-project: don't install the 'sentinel' package yet
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY model/ ${LAMBDA_TASK_ROOT}/model/

# Sync project (installs 'sentinel' package)
RUN uv sync --frozen --no-dev

# Set PATH to use the virtual environment created by uv
ENV PATH="${LAMBDA_TASK_ROOT}/.venv/bin:$PATH"

# Set the CMD to your handler
CMD [ "src.api.handler" ]
