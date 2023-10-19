
#!/bin/bash

if [[ ! -z "${HF_TOKEN}" ]]; then
    echo "The HF_TOKEN environment variable set, logging to Hugging Face."
    python3 -c "import huggingface_hub; huggingface_hub.login('hf_XXXXXXXXXXXXXXXXXXXXXXXXX')"
else
    echo "The HF_TOKEN environment variable is not set or empty, not logging to Hugging Face."
fi

#CMD echo "Y" | ray start --head && sleep 5 && ray status && python -m vllm.entrypoints.openai.api_server \
#        --served-model $MODEL_NAME \
#        --model $MODEL_NAME \
#        --tensor-parallel-size 2 \
#        --worker-use-ray \
#        --host 0.0.0.0 \
#        --port 8080 \
#        --gpu-memory-utilization 0.45 \
#        --max-num-batched-tokens 32768
#        --quantization awq 

# Run the provided command
exec python3 -u -m vllm.entrypoints.openai.api_server "$@"
