FROM docker.io/pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
ARG TAG=latest
ENV WHEEL_FILENAME="asme-${TAG}-py3-none-any.whl"

# install some basic utilities
RUN apt-get update && apt-get -y install zsh byobu tmux curl htop vim git wget mc build-essential xz-utils rsync  && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

#RUN pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY dist/$WHEEL_FILENAME /
RUN pip install --no-cache-dir /$WHEEL_FILENAME

COPY /k8s/release/entrypoint.sh /entrypoint.sh

RUN chmod ugo+rx /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
