# Base image
# The following docker base image is recommended by VLLM: 
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set the working directory
WORKDIR /

# Update and upgrade the system packages (Worker Template)
ARG DEBIAN_FRONTEND=noninteractive

# Install supported GCC version
RUN apt-get update && apt-get -y upgrade
#    apt-get install -y gcc-11 g++-11 && \
#    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 50 --slave /usr/bin/g++ g++ /usr/bin/g++-11

RUN pip install --upgrade pip

COPY builder/setup.sh /setup.sh
RUN chmod +x /setup.sh
RUN bash setup.sh 
RUN rm /setup.sh

# Set CUDA environment variables
ENV cuda_home=/usr/local/cuda-11.8
ENV PATH=${cuda_home}/bin:$PATH
ENV LD_LIBRARY_PATH=${cuda_home}/lib64:$LD_LIBRARY_PATH

RUN echo "$(pip list | grep torch)"
RUN echo "$(python -c 'import torch; print(torch.version.cuda)')"

# TODO: change huggingface_hub when hotfix is released
COPY requirements.txt requirements.txt
RUN pip install requirements.txt

RUN echo "$(pip list | grep torch)"
RUN echo "$(python -c 'import torch; print(torch.version.cuda)')"

# Add src files (Worker Template)
ADD src .  
RUN chmod +x ./benchmark.py && \
    chmod +x ./download_model.py && \
    chmod +x ./handler.py && \
    chmod +x ./metrics.py && \
    chmod +x ./templates.py && \
    chmod +x ./entrypoint.sh

# Prepare the models inside the docker image
ARG HUGGING_FACE_HUB_TOKEN=
ENV HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN
ENV HF_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Prepare argument for the model and tokenizer
ARG MODEL_NAME='Weni/WeniGPT-QA-Mixstral-7B-5.0.0-KTO-AWQ'
ENV MODEL_NAME=$MODEL_NAME
ARG MODEL_REVISION="main"
ENV MODEL_REVISION=$MODEL_REVISION
ARG MODEL_BASE_PATH="/runpod-volume/"
ENV MODEL_BASE_PATH=$MODEL_BASE_PATH
ARG TOKENIZER='Weni/WeniGPT-QA-Mixstral-7B-5.0.0-KTO-AWQ'
ENV TOKENIZER=$TOKENIZER
ARG STREAMING='false'
ENV STREAMING=$STREAMING
ARG DOWNLOAD_MODEL

ENV HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

# Download the models
RUN mkdir -p /model

COPY docker-entrypoint.sh docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh

# Set environment variables
ENV PORT=80 \
    MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    MODEL_BASE_PATH=$MODEL_BASE_PATH \
    HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Conditionally download the model weights based on DOWNLOAD_MODEL
RUN if [ "$DOWNLOAD_MODEL" = "1" ]; then \
    python -u /download_model.py; \
  fi

EXPOSE 8000 6379 80


# ENTRYPOINT ["bash docker-entrypoint.sh"]

# Start the handler
#CMD STREAMING=$STREAMING MODEL_NAME=$MODEL_NAME MODEL_BASE_PATH=$MODEL_BASE_PATH TOKENIZER=$TOKENIZER python -u /handler.py 

#ENTRYPOINT ["./entrypoint.sh"]
#CMD ["--model", "KaleDivergence/WeniGPT-L-70-AWQ-NO-SAFETENSORS", "--host", "0.0.0.0", "--port", "8000", "--gpu-memory-utilization", "0.95", "--tensor-parallel-size", "1", "--tokenizer-mode", "auto", "--seed", "0", "--quantization", "awq", "--dtype", "half"]