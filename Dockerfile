FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -c "import uvicorn; print('uvicorn', uvicorn.__version__)"

COPY . .

# Train Elo + ML once during image build. Runtime requests only load these
# bundles, so /api/analyze and /api/scan-one no longer spend minutes fitting.
RUN python build_model_cache.py

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "weekly_entrypoint:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
