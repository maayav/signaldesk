FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps . && useradd --create-home --uid 10001 appuser
COPY app ./app
COPY .streamlit ./.streamlit
COPY artifacts ./artifacts
RUN mkdir -p /app/var && chown -R appuser:appuser /app/var
USER appuser
EXPOSE 8000 8501
CMD ["uvicorn", "signaldesk.api:app", "--host", "0.0.0.0", "--port", "8000"]
