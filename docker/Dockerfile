FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git

RUN pip install ray==1.8.0 && pip install ray[tune] && pip install ray[rllib]

RUN git clone https://github.com/Replicable-MARL/MARLlib.git

WORKDIR /workspace/MARLlib

RUN pip install -e .

RUN pip install icecream && pip install lbforaging==1.0.15 && pip install supersuit && pip install gym==0.21.0 && pip install importlib-metadata==4.13.0

RUN python patch/add_patch.py -y
