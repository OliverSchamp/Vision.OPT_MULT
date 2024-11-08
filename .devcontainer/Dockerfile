# Use the NVIDIA PyTorch container as a base image
FROM nvcr.io/nvidia/pytorch:23.05-py3

# Set environment variables for Python and pip
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    # cmake \
    # curl \
    # git \
    # unzip \
    # vim \
    && rm -rf /var/lib/apt/lists/*

# Install commonly used Python packages
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    setuptools \
    tqdm \
    pillow \
    pyyaml \
    torchmetrics \
    distinctipy \
    exifread \
    scikit-image \
    xmltodict \
    pymupdf \
    watchdog

RUN pip install -U pillow==9.5.0 --force-reinstall

RUN pip install --no-cache-dir --no-deps python-bidi easyocr streamlit blinker
# Clean up package cache to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /home/oliver

COPY ./dependencies/yolov5/requirements.txt /home/oliver/requirements.txt
RUN pip install --no-cache-dir -r /home/oliver/requirements.txt
COPY ./python/olv_primitives/requirements.txt /home/oliver/requirements.txt
RUN pip install --no-cache-dir -r /home/oliver/requirements.txt

# opencv stuff
RUN pip uninstall -y opencv-python-headless
RUN pip install --no-cache-dir opencv-python==4.5.4.58

# copy and install all mono packages
COPY ./python/ /python/
RUN cd /python/olv_primitives && pip install --no-cache-dir .
RUN cd /python/olv_nms && pip install --no-cache-dir .
RUN cd /python/olv_coco && pip install --no-cache-dir .
RUN cd /python/olv_image && pip install --no-cache-dir .
# RUN cd /python/olv_convert && pip install .
RUN cd /python/olv_ops && pip install --no-cache-dir .
RUN cd /python/olv_ai/olv_ai && pip install --no-cache-dir .
RUN cd /python/olv_ai/olv_object_detection && pip install --no-cache-dir .
RUN cd /python/olv_draw && pip install --no-cache-dir .
RUN cd /python/olv_logging && pip install --no-cache-dir .
RUN rm -r /python

# copy whole project

COPY ./projects/Vision.OPT_MULT /home/oliver/Oliver.Mono/projects/Vision.OPT_MULT
RUN pip install /home/oliver/Oliver.Mono/projects/Vision.OPT_MULT/opt_mult

COPY ./projects/Vision.OPT_MULT/.devcontainer/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

EXPOSE 8051

ENTRYPOINT bash entrypoint.sh