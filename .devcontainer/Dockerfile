# Use the NVIDIA PyTorch container as a base image
FROM ubuntu:22.04

# Set environment variables for Python and pip
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install Python 3 and pip
RUN apt update && apt install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Verify Python and pip installation
RUN python3 --version && pip3 --version

# Create a symbolic link for 'python' and 'pip' to use 'python3' and 'pip3'
RUN ln -s /usr/bin/python3 /usr/bin/python || true && \
    ln -s /usr/bin/pip3 /usr/bin/pip || true

# Update and install system dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install commonly used Python packages
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    setuptools \
    tqdm \
    pillow \
    distinctipy \
    exifread \
    scikit-image \
    xmltodict \
    pymupdf \
    watchdog \
    streamlit \
    pydantic

RUN pip install -U pillow==9.5.0 --force-reinstall
RUN apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /home/oliver

RUN pip install torch>=1.7.0 torchvision>=0.8.1 --index-url https://download.pytorch.org/whl/cpu

COPY ./python/olv_primitives/requirements.txt /home/oliver/requirements_primitives.txt
RUN pip install --no-cache-dir -r /home/oliver/requirements_primitives.txt

# opencv stuff
RUN pip uninstall -y opencv-python-headless
RUN pip install --no-cache-dir opencv-python==4.5.4.58

# copy and install all mono packages
COPY ./python/ /python/
RUN cd /python/olv_primitives && pip install --no-cache-dir .
RUN cd /python/olv_coco && pip install --no-cache-dir .
RUN cd /python/olv_image && pip install --no-cache-dir .
RUN cd /python/olv_ai/olv_ai && pip install --no-cache-dir .
RUN cd /python/olv_ai/olv_object_detection && pip install --no-cache-dir .
RUN cd /python/olv_draw && pip install --no-cache-dir .
RUN rm -r /python

COPY ./projects/Vision.OPT_MULT /home/oliver/Oliver.Mono/projects/Vision.OPT_MULT
RUN pip install /home/oliver/Oliver.Mono/projects/Vision.OPT_MULT/opt_mult

COPY ./projects/Vision.OPT_MULT/.devcontainer/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT bash entrypoint.sh