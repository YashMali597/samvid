# Small, reproducible image. Works on Render, Koyeb, Railway, Cloud Run, or your own VPS.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
# Install dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd --create-home appuser
USER appuser

# Most hosts inject $PORT; default to 8000 locally.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn memory_vault.app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
