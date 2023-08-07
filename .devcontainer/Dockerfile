# Container base image
# PyTorch image with Python 3.9
ARG BASE_IMAGE="pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime"
FROM ${BASE_IMAGE}

# Set display variable to enable rendering through system X server.
# Display and rendering only works with running X server.
ENV DISPLAY=host.docker.internal:0.0

# Update and upgrade standard stuff
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git

# Enable opengl to show OpenGL based rendering of environments
RUN apt-get update -y && apt-get install -y python-opengl

# Install GPUtil for GPU system monitoring via ray
RUN pip install GPUtil