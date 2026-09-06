FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -c "import uvicorn; print('uvicorn', uvicorn.__version__)"

# Keep the expensive data/model build in its own Docker layer. UI/API-only
# changes no longer invalidate the five-league ML training cache.
COPY data ./data
COPY modules ./modules
COPY build_parquet_store.py ./build_parquet_store.py
COPY build_model_cache.py ./build_model_cache.py

RUN python build_parquet_store.py \
    && python build_model_cache.py

# Copy the rest of the application after the precomputed assets exist. Docker
# preserves model_cache/ and generated parquet files because COPY does not
# delete files already present in the image layer.
COPY . .

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "weekly_entrypoint:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
