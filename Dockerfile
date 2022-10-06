# Use the tensorflow image as base for CUDA compatibility
FROM tensorflow/tensorflow:latest-gpu

# metainformation
LABEL org.opencontainers.image.version = "0.0.3"
LABEL org.opencontainers.image.authors = "Saurav Maheshkar"
LABEL org.opencontainers.image.source = "https://github.com/SauravMaheshkar/NeRF"
LABEL org.opencontainers.image.licenses = "MIT"
LABEL org.opencontainers.image.base.name = "index.docker.io/tensorflow/tensorflow:latest-gpu"

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# Helpers
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /code

USER root
RUN apt-get -y install htop

# Essential Installs
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && apt-get install -y --no-install-recommends \
                build-essential \
                gcc \
                gfortran \
                libopenblas-dev \
                ffmpeg \
                libsm6 \
                libxext6 \
                python3 \
                python3.9 \
                python3-pip \
                python3.9-dev \
                python3.9-venv \
                && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /code
RUN python3.9 -m pip install --no-cache-dir --upgrade pip setuptools wheel isort networkx scipy
RUN python3.9 -m pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt