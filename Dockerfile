FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/images \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser final_backend.py final_frontend.py ./

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "final_frontend.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
