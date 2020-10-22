# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:20.04

MAINTAINER Rocketmiles <eric@rocketmiles.com>

RUN apt-get -y update && \
    apt-get -y install software-properties-common && \
    apt-get -y update && \
    apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
         build-essential \
         libssl-dev \
         libffi-dev \
         libxml2-dev \
         libxslt1-dev \
         zlib1g-dev \
         python3-pip \
         python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Here we get all python packages.
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt && \
    rm -rf /root/.cache

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

WORKDIR "/root/"