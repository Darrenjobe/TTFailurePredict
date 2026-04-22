FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc libpq-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY feature_specs ./feature_specs
COPY sql ./sql
COPY scripts ./scripts

RUN pip install -e .

EXPOSE 8080 8501

CMD ["uvicorn", "survpredict.inference.service:app", "--host", "0.0.0.0", "--port", "8080"]
