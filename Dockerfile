FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
    && apt-get install --yes --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy==2.4.6 pillow==12.3.0 matplotlib==3.11.2 openslide-python==1.4.6 openslide-bin==4.0.1.2

COPY pyproject.toml /opt/stainnorm/
COPY stainnorm /opt/stainnorm/stainnorm
RUN pip install --no-deps /opt/stainnorm
