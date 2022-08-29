FROM nvidia/cuda:10.1-devel-ubuntu18.04

# the uid defined in .env and passed from docker-compose build arg
ARG uid

# avoid unable to initialize frontend errors
ENV DEBIAN_FRONTEND noninteractive

# install apt-utils to allow package configuration
RUN apt-get update && apt-get install -y apt-utils

# install generic build dependencies
RUN apt-get update && apt-get install -y wget vim git curl jq nano

# install python
RUN apt-get update && apt-get install -y python3.7 python3.7-dev

# install python package manager tools
RUN apt-get update && apt-get install -y python3-pip

# Install python requirements
COPY requirements.txt ./
RUN pip3 install -U pip
RUN pip3 install --no-cache-dir -r requirements.txt
# Install PyTorch Version for Cuda 10.1
RUN pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html


# Clean up
RUN apt-get -q autoremove &&\
    apt-get -q clean -y && rm -rf /var/lib/apt/lists/* && rm -f /var/cache/apt/*.bin

WORKDIR /app/main