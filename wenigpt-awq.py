# Need to run 
#pip install autoawq
# first

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

model_path = 'Weni/WeniGPT-L-70'
quant_path = 'WeniGPT-L-70-AWQ'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
# NOTE: pass safetensors=True to load safetensors
model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model as safetensors
model.save_quantized(quant_path, safetensors=True)

# Save quantized model as pytorch binary
model.save_quantized(quant_path)

tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved (in both binary and safetensors format) at "{quant_path}"')

