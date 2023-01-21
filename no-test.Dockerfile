# FROM python:3.8.6-slim
# https://hub.docker.com/r/nvidia/cuda/
# FROM nvidia/cuda:11.1.1-devel
# FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
# NOTE:
# pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime has python version 3.10.8
# pytorch/pytorch:1.10.1-cuda11.3-cudnn8-runtime has python version 3.7.x

# NOTE: The GPU provided by the Challenge is GPU Tesla T4 with nvidiaDriverVersion: 470.82.01
# by checking https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
# and https://download.pytorch.org/whl/torch_stable.html

## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"

## Install your dependencies here using apt install, etc.

# latest version of biosppy uses opencv
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt update
RUN apt install build-essential -y
RUN apt install git ffmpeg libsm6 libxext6 vim libsndfile1 -y

RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

# alternative pypi sources
# http://mirrors.aliyun.com/pypi/simple/
# http://pypi.douban.com/simple/
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -m pip install --upgrade pip setuptools wheel

# NOTE that torch and torchaudio should be installed first
# torch already installed in the base image
# RUN pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
# compatible with torch
RUN pip install torchaudio==0.13.1+cu116 --no-deps -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install torch-ecg
RUN pip install torch-audiomentations --no-deps

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./requirements-no-torch.txt /physionet
WORKDIR /physionet

RUN pip install -r requirements-no-torch.txt

COPY ./ /physionet


# NOTE: also run test_local.py to test locally
# since GitHub Actions does not have GPU,
# one need to run test_local.py to avoid errors related to devices
# RUN python test_docker.py


# commands to run test with docker container:

# sudo docker build -t image .
# sudo docker run -it --shm-size=10240m --gpus all -v ~/Jupyter/temp/cinc2022_docker_test/model:/physionet/model -v ~/Jupyter/temp/cinc2022_docker_test/test_data:/physionet/test_data -v ~/Jupyter/temp/cinc2022_docker_test/test_outputs:/physionet/test_outputs -v ~/Jupyter/temp/cinc2022_docker_test/data:/physionet/training_data image bash


# python train_model.py training_data model
# python test_model.py model test_data test_outputs
