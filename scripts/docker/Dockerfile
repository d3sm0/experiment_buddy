# -*- mode: dockerfile -*-
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

ARG PYTHON_VERSION=3.8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq \
        build-essential \
        curl \
        git \
        ninja-build \
        libglib2.0-0 \
        sudo\
        wget \
        python3-opencv\
        python3 \
        python3-pip\
        python3.8-venv\
    && rm -rf /var/lib/apt/lists/*


WORKDIR /opt

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip install --upgrade pip
RUN pip install numpy torch --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir
RUN pip install git+https://github.com/d3sm0/experiment_buddy#egg=experiment_buddy --no-cache-dir
ENV BUDDY_IS_DEPLOYED=1

RUN touch /opt/entrypoint.sh && chmod 0755 /opt/entrypoint.sh
COPY scripts/docker/entrypoint.sh /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]
