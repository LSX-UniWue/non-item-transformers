ARG BASE_IMAGE=docker.io/fedora:35
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=1.11.0
ARG PYTORCH_VISION_VERSION=0.12.0
ARG PYTORCH_TEXT_VERSION=0.12.0
ARG PYTORCH_AUDIO_VERSION=0.11.0

FROM docker.io/fedora:35 as asme-build
RUN dnf makecache && dnf install -y poetry && dnf clean all
COPY . /asme
RUN cd /asme && poetry build


FROM ${BASE_IMAGE} as dev-base
RUN dnf makecache -y && \
    dnf install -y \
        make \
        automake \
        gcc \
        gcc-c++ \
        kernel-devel \
        ca-certificates \
        cmake \
        curl \
        git \
        libjpeg-turbo-devel \
        libpng-devel && \
    dnf clean all
ENV PATH /opt/conda/bin:$PATH


FROM dev-base as conda
ARG PYTHON_VERSION=3.10
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython && \
    /opt/conda/bin/conda clean -ya


FROM conda as conda-installs
ARG CUDA_VERSION=11.3
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=1.11.0
ARG PYTORCH_VISION_VERSION=0.12.0
ARG PYTORCH_TEXT_VERSION=0.12.0
ARG PYTORCH_AUDIO_VERSION=0.11.0
ARG PYTORCH_LIGHTNING_VERSION=1.6.3
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}
RUN /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y "python=${PYTHON_VERSION}" "pytorch=${PYTORCH_VERSION}" "torchvision=${PYTORCH_VISION_VERSION}" "torchtext=${PYTORCH_TEXT_VERSION}" "torchaudio=${PYTORCH_AUDIO_VERSION}" "cudatoolkit=${CUDA_VERSION}" && \
    /opt/conda/bin/conda clean -ya \
RUN /opt/conda/bin/conda install -c conda-forge -y "pytorch-lightning=${PYTORCH_LIGHTNING_VERSION}" "torchmetrics" "scipy" "pandas" "scikit-learn" "optuna" "matplotlib" "seaborn" "tqdm" "rich" && \
    /opt/conda/bin/conda clean -ya
RUN /opt/conda/bin/pip install torchelastic


FROM ${BASE_IMAGE} as official
ARG PYTORCH_VERSION
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN dnf makecache && dnf install -y \
        ca-certificates \
        libjpeg-turbo-devel \
        libpng-devel && \
    dnf clean all
COPY --from=conda-installs /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION ${PYTORCH_VERSION}
WORKDIR /workspace


FROM official as asme-release
RUN dnf makecache && dnf install -y \
        zsh \
        byobu \
        tmux \
        curl \
        htop \
        vim \
        git \
        wget \
        rsync \
        mc \
        make \
        automake \
        gcc \
        gcc-c++ \
        xz && \
    dnf clean all
ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
COPY --from=asme-build /asme/dist/asme-1.0.0-py3-none-any.whl /
RUN /opt/conda/bin/pip install --no-cache-dir /asme-1.0.0-py3-none-any.whl
RUN rm /asme-1.0.0-py3-none-any.whl
COPY /k8s/release/entrypoint.sh /entrypoint.sh

RUN chmod ugo+rx /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]


FROM official as asme-dev
RUN dnf makecache && dnf install -y \
        zsh \
        byobu \
        tmux \
        curl \
        htop \
        vim \
        git \
        wget \
        rsync \
        mc \
        make \
        automake \
        gcc \
        gcc-c++ \
        xz && \
    dnf clean all
# install poetry into /opt/poetry
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH="/opt/poetry/bin:${PATH}"
RUN chmod o+rx /opt/poetry/bin/poetry

ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV REPO_USER=dallmann
ENV REPO_BRANCH=master
ENV PROJECT_DIR=/project

COPY k8s/dev/.git-askpass /.git-askpass
RUN chmod +x /.git-askpass

ENV GIT_TOKEN="123456"
ENV GIT_ASKPASS=/.git-askpass

COPY k8s/dev/entrypoint.sh /entrypoint.sh
RUN chmod ugo+rx /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh" ]
