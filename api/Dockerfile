FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY detection-engine ./detection-engine
COPY app ./app
RUN mkdir -p detection-engine/output

ENV PYTHONPATH=/app/detection-engine

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
