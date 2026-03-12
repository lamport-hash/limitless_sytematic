FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -e . || pip install --no-cache-dir -r requirements.txt

COPY ui_fastapi/requirements.txt ./ui_fastapi/
RUN pip install --no-cache-dir -r ui_fastapi/requirements.txt

COPY . .

WORKDIR /app/ui_fastapi

EXPOSE 8000 8888 8890

CMD ["python", "app.py"]
