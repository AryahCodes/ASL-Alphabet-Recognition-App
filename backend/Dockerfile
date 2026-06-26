FROM python:3.11.9-slim-bookworm
WORKDIR /app
ENV SIGNAPP_INFERENCE_BACKEND=numpy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN groupadd --system signapp \
    && useradd --system --gid signapp --create-home signapp
COPY --chown=signapp:signapp . .
USER signapp
EXPOSE 5001
CMD gunicorn --worker-class eventlet -w 1 --timeout 120 --graceful-timeout 30 -b 0.0.0.0:${PORT:-5001} server:app
