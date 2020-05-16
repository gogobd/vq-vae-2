FROM nvidia/cuda:latest

# Install system dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        curl \
        wget \
        git \
        unzip \
        screen \
        vim \
        net-tools \
    && apt-get clean

# Install python miniconda3 + requirements
ENV MINICONDA_HOME /opt/miniconda
ENV PATH ${MINICONDA_HOME}/bin:${PATH}
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x Miniconda3-latest-Linux-x86_64.sh \
    && ./Miniconda3-latest-Linux-x86_64.sh -b -p "${MINICONDA_HOME}" \
    && rm Miniconda3-latest-Linux-x86_64.sh
RUN conda update -n base -c defaults conda

# JupyterLab
RUN conda install -c conda-forge jupyterlab ipywidgets nodejs

# Project
COPY . /vq-vae-2
WORKDIR /vq-vae-2
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y

# Start container in notebook mode
CMD SHELL=/bin/bash jupyter lab --no-browser --ip 0.0.0.0 --port 8888 --allow-root

# docker build -t vq-vae-2 .
# docker run -v /host/directory/data:/data -p 8888:8888 --ipc=host --gpus all -it vq-vae-2
