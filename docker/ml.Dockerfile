FROM python:3.13-slim

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY src/ml_api ./src/ml_api
COPY config ./config
COPY models ./models

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "src.ml_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
