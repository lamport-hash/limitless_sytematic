FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY visualiser_app /app/visualiser_app

EXPOSE 8000

ENV PYTHONPATH=/app
ENV BUNDLE_DIR=/app/data/bundle

CMD ["python", "-m", "uvicorn", "visualiser_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
