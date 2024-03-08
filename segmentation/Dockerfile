FROM python:3.8-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git \
    && apt-get install -qy --no-install-recommends wget git libopencv-dev python3-opencv \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=True \
    PORT=7070 \
    WORKERS=2 \
    THREADS=4

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

ENV GOOGLE_APPLICATION_CREDENTIALS=../secrets/service_account_key.json

CMD exec gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
