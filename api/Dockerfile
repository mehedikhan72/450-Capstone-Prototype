FROM python:3.11-slim

WORKDIR /app

# LightGBM links against the OpenMP runtime, which python:*-slim does not ship.
# Without it `import lightgbm` dies with:
#   OSError: libgomp.so.1: cannot open shared object file
# and the TRIDENT bundle cannot be unpickled at all, because its 'Boosting'
# member is an LGBMClassifier.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY detection-engine ./detection-engine
COPY app ./app
RUN mkdir -p detection-engine/output

ENV PYTHONPATH=/app/detection-engine

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
