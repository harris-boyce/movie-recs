FROM mcr.microsoft.com/devcontainers/anaconda:3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with ML dependencies
COPY requirements-dev.txt /tmp/
RUN conda create -n mlenv python=3.10 -y && \
    echo "conda activate mlenv" >> ~/.bashrc && \
    /opt/conda/envs/mlenv/bin/pip install -r /tmp/requirements-dev.txt

# Set conda environment as default
ENV PATH=/opt/conda/envs/mlenv/bin:$PATH
ENV CONDA_DEFAULT_ENV=mlenv

# Install additional tools
RUN /opt/conda/envs/mlenv/bin/pip install \
    jupyter \
    jupyterlab \
    mlflow \
    wandb \
    tensorboard

# Create directories
RUN mkdir -p /mlflow /models /data /workspace

# Set working directory
WORKDIR /workspace

# Expose ports
EXPOSE 8888 5000 8080