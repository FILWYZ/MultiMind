



FROM python:3.12-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app


FROM base AS dependencies

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


FROM base AS production


RUN useradd -m -u 1000 echomind


COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

COPY --chown=echomind:echomind . .


RUN mkdir -p /app/data/chroma /app/logs /app/config && \
    chown echomind:echomind /app/data /app/data/chroma /app/logs /app/config
USER echomind

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5).read()"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM dependencies AS development

COPY . .

RUN mkdir -p /app/data/chroma /app/logs /app/config /app/tests && \
    chmod -R 777 /app/data /app/logs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
