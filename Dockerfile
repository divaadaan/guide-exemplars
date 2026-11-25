# guide-exemplars API + dashboard.
#   docker compose up --build              -> http://localhost:8000
#   docker build --build-arg LOCAL_EMBED=true .   # adds sentence-transformers (CPU torch) for KNN-SBERT
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

ARG LOCAL_EMBED=false
RUN if [ "$LOCAL_EMBED" = "true" ]; then \
        uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu && \
        uv pip install --system ".[llm,data,service,local-embed]"; \
    else \
        uv pip install --system ".[llm,data,service]"; \
    fi

COPY configs ./configs

ENV GUIDE_CONFIG_DIR=/app/configs \
    GUIDE_DATA_DIR=/data \
    GUIDE_RESULTS_DIR=/results \
    GUIDE_CACHE_DIR=/cache \
    GUIDE_DB_PATH=/cache/service.db

RUN useradd --create-home --uid 1000 app && mkdir -p /data /results /cache && chown app /data /results /cache
USER app
WORKDIR /home/app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status == 200 else 1)"

CMD ["guide", "serve", "--host", "0.0.0.0", "--port", "8000"]
