FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY ui_strat/requirements.txt ./
RUN uv pip install --system -r requirements.txt

COPY core/ ./core/
COPY features/ ./features/
COPY strat/ ./strat/
COPY backtest/ ./backtest/
COPY norm/ ./norm/
COPY ui_strat/ ./ui_strat/

ENV PYTHONPATH=/app

WORKDIR /app/ui_strat

RUN mkdir -p static/charts templates

EXPOSE 8500

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8500"]
