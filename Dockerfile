# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec first (layer caching)
COPY pyproject.toml ./

# Phase 1: install dependencies only (cached unless pyproject.toml changes).
# Stub source tree satisfies setuptools package discovery during metadata build.
RUN touch README.md && mkdir -p src/rag && touch src/rag/__init__.py \
    && pip install --no-cache-dir ".[openai,qdrant,distributed,scaledown]"

# Copy real source code (after deps so source changes don't bust the cache)
COPY src/ src/
COPY scripts/ scripts/
COPY settings.toml ./

# Phase 2: reinstall project only (no deps) so site-packages has real code
RUN pip install --no-cache-dir --no-deps .

# ---- Runtime stage ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app/src src/
COPY --from=builder /app/scripts scripts/
COPY --from=builder /app/settings.toml settings.toml
COPY --from=builder /app/pyproject.toml pyproject.toml

# Copy entrypoint
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create artifacts directory
RUN mkdir -p /app/artifacts

# Run as non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["help"]
