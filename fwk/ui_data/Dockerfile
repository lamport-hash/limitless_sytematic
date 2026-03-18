FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY ui_data/requirements.txt ./
RUN uv pip install --system -r requirements.txt

COPY core/ ./core/
COPY features/ ./features/
COPY merger/ ./merger/
COPY bundler/ ./bundler/
COPY norm/ ./norm/
COPY ui_data/ ./ui_data/

EXPOSE 8765

CMD ["python", "-m", "uvicorn", "ui_data.main:app", "--host", "0.0.0.0", "--port", "8765"]
