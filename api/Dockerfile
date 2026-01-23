# Use Python slim image as base
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=false \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app


# ============================================
# Dependency installation layer (cached unless pyproject.toml or uv.lock changes)
# ============================================
FROM base AS dependencies

# Copy only dependency files first for better caching
COPY pyproject.toml uv.lock* ./

# Install dependencies into a virtual environment
# This layer will be cached unless your dependencies change
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ============================================
# Development stage with dev dependencies
# ============================================
FROM dependencies AS development

# Install dev dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy application code
COPY . .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

CMD ["uv", "run", "uvicorn", "livinggo.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================
# Production stage (optimized, no dev dependencies)
# ============================================
FROM dependencies AS production

# Copy application code
COPY . .

# Install the project itself (no dev dependencies)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "livinggo.main"]