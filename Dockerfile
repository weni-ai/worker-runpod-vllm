# Base image
# The following docker base image is recommended by VLLM: 
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set the working directory
WORKDIR /

# Update and upgrade the system packages (Worker Template)
ARG DEBIAN_FRONTEND=noninteractive

# Install supported GCC version
#RUN apt-get update && \
#    apt-get install -y gcc-11 g++-11 && \
#    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 50 --slave /usr/bin/g++ g++ /usr/bin/g++-11

RUN pip install --upgrade pip
#RUN pip uninstall torch -y
#RUN pip install torch==2.0.1 -f https://download.pytorch.org/whl/cu118
COPY builder/setup.sh /setup.sh
RUN chmod +x /setup.sh && \
    /setup.sh && \
    rm /setup.sh

# Set CUDA environment variables
ENV cuda_home=/usr/local/cuda-11.8
ENV PATH=${cuda_home}/bin:$PATH
ENV LD_LIBRARY_PATH=${cuda_home}/lib64:$LD_LIBRARY_PATH

RUN echo "$(pip list | grep torch)"
RUN echo "$(python -c 'import torch; print(torch.version.cuda)')"

# If HF_MODEL_QUANTIZE="gptq" then install vllm (GPTQ Fork) from the source
# Otherwise if HF_MODEL_QUANTIZE="awq" then install vllm==0.2.0 from PyPI
# Else install vllm==0.2.0 from PyPI
RUN pip install fastapi==0.99.1 \
        vllm==0.2.0 \
        huggingface-hub==0.16.4 \
        runpod==1.2.1

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

# Quick temporary updates
RUN pip install git+https://github.com/runpod/runpod-python@a1#egg=runpod --compile

# Prepare the models inside the docker image
ARG HUGGING_FACE_HUB_TOKEN='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'
ENV HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN
ENV HF_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Prepare argument for the model and tokenizer
ARG MODEL_NAME='KaleDivergence/WeniGPT-L-70-AWQ-NO-SAFETENSORS'
ENV MODEL_NAME=$MODEL_NAME
ARG MODEL_REVISION="main"
ENV MODEL_REVISION=$MODEL_REVISION
ARG MODEL_BASE_PATH="/runpod-volume/"
ENV MODEL_BASE_PATH=$MODEL_BASE_PATH
ARG TOKENIZER='KaleDivergence/WeniGPT-L-70-AWQ-NO-SAFETENSORS'
ENV TOKENIZER=$TOKENIZER
ARG STREAMING='false'
ENV STREAMING=$STREAMING

ENV HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

# Download the models
RUN mkdir -p /model

# Set environment variables
ENV PORT=80 \
    MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    MODEL_BASE_PATH=$MODEL_BASE_PATH \
    HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Run the Python script to download the model
#RUN python -u /download_model.py

# Start the handler
#CMD STREAMING=$STREAMING MODEL_NAME=$MODEL_NAME MODEL_BASE_PATH=$MODEL_BASE_PATH TOKENIZER=$TOKENIZER python -u /handler.py 

EXPOSE 8080 6379 80

#        --served-model $MODEL_NAME \
#        --model $MODEL_NAME \
#        --tensor-parallel-size 2 \
#        --worker-use-ray \
#        --host 0.0.0.0 \
#        --port 8080 \
#        --gpu-memory-utilization 0.45 \
#        --max-num-batched-tokens 32768
ENTRYPOINT ["./entrypoint.sh"]
#CMD ["--model", "/runpod-volume/WeniGPT-L-70-AWQ-NO-SAFETENSORS", "--host", "0.0.0.0", "--port", "8080", "--gpu-memory-utilization", "0.45", "--tensor-parallel-size", "2", "--tokenizer_mode", "auto", "--max_num_batched_tokens", "8192", "--max_model_len", "4094", "--seed", "0", "--dtype", "auto"]
CMD ["--model", "KaleDivergence/WeniGPT-L-70-AWQ-NO-SAFETENSORS", "--host", "0.0.0.0", "--port", "8080", "--gpu-memory-utilization", "0.95", "--tensor-parallel-size", "1", "--tokenizer-mode", "auto", "--seed", "0", "--quantization", "awq", "--dtype", "half"]
