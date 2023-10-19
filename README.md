<div align="center">

<h1>WeniGPT vLLM Endpoint | Runpod Serverless Worker </h1>

[![CI | Test Worker](https://github.com/matthew-mcateer/worker-runpod-vllm/actions/workflows/CI-test_worker.yml/badge.svg)](https://github.com/matthew-mcateer/worker-runpod-vllm/actions/workflows/CI-test_worker.yml)
&nbsp;
[![Docker Image](https://github.com/matthew-mcateer/worker-runpod-vllm/actions/workflows/CD-docker_dev.yml/badge.svg)](https://github.com/matthew-mcateer/worker-runpod-vllm/actions/workflows/CD-docker_dev.yml)

🚀 | This serverless worker utilizes vLLM (very Large Language Model) behind the scenes and is integrated into RunPod's serverless environment. It supports dynamic auto-scaling using the built-in RunPod autoscaling feature.
</div>

#### Docker Arguments:
1. `HUGGING_FACE_HUB_TOKEN`: Your private Hugging Face token. This token is required for downloading models that necessitate agreement to an End User License Agreement (EULA), such as the llama2 family of models.
2. `MODEL_NAME`: The Hugging Face model to use. Please ensure that the chosen model is supported by vLLM. Refer to the list of supported models for compatibility.
3. `TOKENIZER`: (Optional) The specified tokenizer to use. If you want to use the default tokenizer for the model, do not provide this docker argument at all.
4. `STREAMING`: Whether to use HTTP Streaming or not. Specify True if you want to enable HTTP Streaming; otherwise, omit this argument.

#### llama2 7B Chat:

`docker build . --platform linux/amd64 --build-arg HUGGING_FACE_HUB_TOKEN=your_hugging_face_token_here --build-arg MODEL_NAME=meta-llama/Llama-2-7b-chat-hf --build-arg TOKENIZER=hf-internal-testing/llama-tokenizer --build-arg STREAMING=True`

#### llama2 13B Chat:

`docker build . --platform linux/amd64 --build-arg HUGGING_FACE_HUB_TOKEN=your_hugging_face_token_here --build-arg MODEL_NAME=meta-llama/Llama-2-13b-chat-hf --build-arg TOKENIZER=hf-internal-testing/llama-tokenizer --build-arg STREAMING=True`

#### WeniGPT 70B 4-bit:

`docker build . -t matthewmcateer0/worker-runpod-vllm:deploy --progress=plain --no-cache --platform linux/amd64 --build-arg HUGGING_FACE_HUB_TOKEN=your_hugging_face_token_here --build-arg MODEL_NAME=KaleDivergence/WeniGPT-L-70-AWQ-NO-SAFETENSORS --build-arg TOKENIZER=KaleDivergence/WeniGPT-L-70-AWQ-NO-SAFETENSORS --build-arg STREAMING=False`

<!--

docker build . -t matthewmcateer0/worker-runpod-vllm:wenigpt70b4bit-vllm0.2.0 --progress=plain --no-cache --platform linux/amd64 --build-arg HF_MODEL_ID="Weni/WeniGPT-L-70-4bit" --build-arg HF_MODEL_REVISION="main" --build-arg SM_NUM_GPUS="1" --build-arg HF_MODEL_QUANTIZE="gptq" --build-arg HF_MODEL_TRUST_REMOTE_CODE="true" --build-arg HUGGING_FACE_HUB_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXX" --build-arg HF_MAX_TOTAL_TOKENS="10240" --build-arg HF_MAX_INPUT_LENGTH="8192" --build-arg HF_MAX_BATCH_TOTAL_TOKENS="10240" --build-arg HF_MAX_BATCH_PREFILL_TOKENS="8192" --build-arg DOWNLOAD_MODEL="1"
-->

Five images:
- `matthewmcateer0/worker-runpod-vllm:wenigpt70bawq-vllm0.2.0`
- `matthewmcateer0/worker-runpod-vllm:wenigpt70b4bit-vllm0.1.7gptq`
- `matthewmcateer0/worker-runpod-vllm:wenigpt70b4bit-vllm0.2.0`

- `matthewmcateer0/worker-runpod-vllm:Mistral7BAWQ-vllm0.2.0`
- `matthewmcateer0/worker-runpod-vllm:Mistral7B-vllm0.2.0`


The alternate image:
- `matthewmcateer0/worker-runpod-vllm:dev`

```bash
Launching vLLM with args: --model TheBlokeAI/Test-AWQ-13B-128-No_Safetensors --quantization awq --host 0.0.0.0
INFO 09-17 14:15:33 llm_engine.py:72] Initializing an LLM engine with config: model='TheBlokeAI/Test-AWQ-13B-128-No_Safetensors', tokenizer='TheBlokeAI/Test-AWQ-13B-128-No_Safetensors', tokenizer_mode=auto, revision=None, trust_remote_code=False, dtype=torch.float16, download_dir=None, load_format=auto, tensor_parallel_size=1, quantization=awq, seed=0)
INFO 09-17 14:17:03 llm_engine.py:202] # GPU blocks: 2600, # CPU blocks: 327
INFO:     Started server process [38]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```


Weni/WeniGPT-L-70-4bit (_without_ GPTQ-optimized vLLM), ``:
```bash
sudo docker build  . -t matthewmcateer0/worker-runpod-vllm:wenigpt70b4bit-vllm0.2.0 --progress=plain --no-cache --platform linux/amd64  --build-arg HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXX --build-arg HF_MODEL_QUANTIZE=none --build-arg MODEL_NAME=Weni/WeniGPT-L-70-4bit --build-arg TOKENIZER=hf-internal-testing/llama-tokenizer --build-arg STREAMING=False --build-arg DOWNLOAD_MODEL=1

sudo docker push matthewmcateer0/worker-runpod-vllm:wenigpt70b4bit-vllm0.2.0
```
```
2023-10-11T17:49:47.105699763Z Traceback (most recent call last):
2023-10-11T17:49:47.105711003Z   File "/handler.py", line 50, in <module>
2023-10-11T17:49:47.105811203Z     llm = AsyncLLMEngine.from_engine_args(engine_args)
2023-10-11T17:49:47.105813843Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 486, in from_engine_args
2023-10-11T17:49:47.105918034Z     engine = cls(engine_args.worker_use_ray,
2023-10-11T17:49:47.105920714Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 270, in __init__
2023-10-11T17:49:47.105977864Z     self.engine = self._init_engine(*args, **kwargs)
2023-10-11T17:49:47.105979534Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 306, in _init_engine
2023-10-11T17:49:47.106054944Z     return engine_class(*args, **kwargs)
2023-10-11T17:49:47.106056964Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py", line 108, in __init__
2023-10-11T17:49:47.106087804Z     self._init_workers(distributed_init_method)
2023-10-11T17:49:47.106089404Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py", line 140, in _init_workers
2023-10-11T17:49:47.106127264Z     self._run_workers(
2023-10-11T17:49:47.106129784Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py", line 692, in _run_workers
2023-10-11T17:49:47.106239104Z     output = executor(*args, **kwargs)
2023-10-11T17:49:47.106240955Z   File "/usr/local/lib/python3.10/dist-packages/vllm/worker/worker.py", line 68, in init_model
2023-10-11T17:49:47.106277775Z     self.model = get_model(self.model_config)
2023-10-11T17:49:47.106279325Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/model_loader.py", line 91, in get_model
2023-10-11T17:49:47.106315355Z     model = model_class(model_config.hf_config, quant_config)
2023-10-11T17:49:47.106316995Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/llama.py", line 278, in __init__
2023-10-11T17:49:47.106396855Z     self.model = LlamaModel(config, quant_config)
2023-10-11T17:49:47.106399125Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/llama.py", line 236, in __init__
2023-10-11T17:49:47.106459905Z     self.layers = nn.ModuleList([
2023-10-11T17:49:47.106461725Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/llama.py", line 237, in <listcomp>
2023-10-11T17:49:47.106532025Z     LlamaDecoderLayer(config, quant_config)
2023-10-11T17:49:47.106533945Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/llama.py", line 182, in __init__
2023-10-11T17:49:47.106580205Z     self.mlp = LlamaMLP(
2023-10-11T17:49:47.106582896Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/llama.py", line 69, in __init__
2023-10-11T17:49:47.106635196Z     self.down_proj = ParallelLinear.row(intermediate_size,
2023-10-11T17:49:47.106636936Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/quantized_linear/__init__.py", line 30, in row
2023-10-11T17:49:47.106664956Z     return RowParallelLinear(*args, **kwargs)
2023-10-11T17:49:47.106666996Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/parallel_utils/tensor_parallel/layers.py", line 313, in __init__
2023-10-11T17:49:47.106741366Z     self.create_weights(params_dtype)
2023-10-11T17:49:47.106743486Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/parallel_utils/tensor_parallel/layers.py", line 331, in create_weights
2023-10-11T17:49:47.106803556Z     self.weight = Parameter(torch.empty(
2023-10-11T17:49:47.106806446Z torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 448.00 MiB (GPU 0; 79.15 GiB total capacity; 78.19 GiB already allocated; 329.25 MiB free; 78.19 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_
```
```
2023-10-11T13:29:41.235283139Z Traceback (most recent call last):
2023-10-11T13:29:41.235317886Z   File "/handler.py", line 44, in <module>
2023-10-11T13:29:41.235434922Z     llm = AsyncLLMEngine.from_engine_args(engine_args)
2023-10-11T13:29:41.235458941Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 486, in from_engine_args
2023-10-11T13:29:41.235464903Z     engine = cls(engine_args.worker_use_ray,
2023-10-11T13:29:41.235483033Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 270, in __init__
2023-10-11T13:29:41.235489229Z     self.engine = self._init_engine(*args, **kwargs)
2023-10-11T13:29:41.235493688Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 306, in _init_engine
2023-10-11T13:29:41.235632002Z     return engine_class(*args, **kwargs)
2023-10-11T13:29:41.235658641Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py", line 108, in __init__
2023-10-11T13:29:41.235667212Z     self._init_workers(distributed_init_method)
2023-10-11T13:29:41.235674515Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py", line 140, in _init_workers
2023-10-11T13:29:41.235681945Z     self._run_workers(
2023-10-11T13:29:41.235690158Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py", line 692, in _run_workers
2023-10-11T13:29:41.235971164Z     output = executor(*args, **kwargs)
2023-10-11T13:29:41.235992789Z   File "/usr/local/lib/python3.10/dist-packages/vllm/worker/worker.py", line 68, in init_model
2023-10-11T13:29:41.235998101Z     self.model = get_model(self.model_config)
2023-10-11T13:29:41.236003157Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/model_loader.py", line 91, in get_model
2023-10-11T13:29:41.236008064Z     model = model_class(model_config.hf_config, quant_config)
2023-10-11T13:29:41.236012441Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/llama.py", line 278, in __init__
2023-10-11T13:29:41.236016851Z     self.model = LlamaModel(config, quant_config)
2023-10-11T13:29:41.236021106Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/llama.py", line 236, in __init__
2023-10-11T13:29:41.236027131Z     self.layers = nn.ModuleList([
2023-10-11T13:29:41.236032229Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/llama.py", line 237, in <listcomp>
2023-10-11T13:29:41.236150543Z     LlamaDecoderLayer(config, quant_config)
2023-10-11T13:29:41.236157540Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/llama.py", line 182, in __init__
2023-10-11T13:29:41.236161900Z     self.mlp = LlamaMLP(
2023-10-11T13:29:41.236166429Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/llama.py", line 63, in __init__
2023-10-11T13:29:41.236170730Z     self.gate_up_proj = ParallelLinear.column(hidden_size,
2023-10-11T13:29:41.236175060Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/quantized_linear/__init__.py", line 17, in column
2023-10-11T13:29:41.236179550Z     return ColumnParallelLinear(*args, **kwargs)
2023-10-11T13:29:41.236183713Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/parallel_utils/tensor_parallel/layers.py", line 198, in __init__
2023-10-11T13:29:41.236189170Z     self.create_weights(params_dtype)
2023-10-11T13:29:41.236193296Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/parallel_utils/tensor_parallel/layers.py", line 213, in create_weights
2023-10-11T13:29:41.236428625Z     self.weight = Parameter(torch.empty(
```


Weni/WeniGPT-L-70-4bit (w/ GPTQ-optimized vLLM), ``:
```bash
sudo docker build  . -t matthewmcateer0/worker-runpod-vllm:wenigpt70b4bit-vllm0.1.7gptq --progress=plain --no-cache --platform linux/amd64  --build-arg HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXX --build-arg HF_MODEL_QUANTIZE=gptq --build-arg MODEL_NAME=Weni/WeniGPT-L-70-4bit --build-arg TOKENIZER=hf-internal-testing/llama-tokenizer --build-arg STREAMING=False --build-arg DOWNLOAD_MODEL=1

sudo docker push matthewmcateer0/worker-runpod-vllm:wenigpt70b4bit-vllm0.1.7gptq
```




Weni/WeniGPT-L-70-4bit

KaleDivergence/WeniGPT-L-70-AWQ, ``:
```bash
sudo docker build  . -t matthewmcateer0/worker-runpod-vllm:wenigpt70bawq-vllm0.2.0 --progress=plain --no-cache --platform linux/amd64  --build-arg HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXX --build-arg HF_MODEL_QUANTIZE=awq --build-arg MODEL_NAME=KaleDivergence/WeniGPT-L-70-AWQ --build-arg TOKENIZER=hf-internal-testing/llama-tokenizer --build-arg STREAMING=False --build-arg DOWNLOAD_MODEL=1

sudo docker push matthewmcateer0/worker-runpod-vllm:wenigpt70bawq-vllm0.2.0
```









TheBloke/Mistral-7B-v0.1-AWQ, ``:
```bash
sudo docker build  . -t matthewmcateer0/worker-runpod-vllm:Mistral7BAWQ-vllm0.2.0 --progress=plain --no-cache --platform linux/amd64  --build-arg HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXX --build-arg HF_MODEL_QUANTIZE=awq --build-arg MODEL_NAME=TheBloke/Mistral-7B-v0.1-AWQ --build-arg TOKENIZER=hf-internal-testing/llama-tokenizer --build-arg STREAMING=False --build-arg DOWNLOAD_MODEL=1

sudo docker push matthewmcateer0/worker-runpod-vllm:Mistral7BAWQ-vllm0.2.0
```
```
2023-10-11T09:34:21.454481787Z Traceback (most recent call last):
2023-10-11T09:34:21.454500994Z   File "/handler.py", line 50, in <module>
2023-10-11T09:34:21.454504530Z     llm = AsyncLLMEngine.from_engine_args(engine_args)
2023-10-11T09:34:21.454507987Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 486, in from_engine_args
2023-10-11T09:34:21.454578421Z     engine = cls(engine_args.worker_use_ray,
2023-10-11T09:34:21.454582739Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 270, in __init__
2023-10-11T09:34:21.454585514Z     self.engine = self._init_engine(*args, **kwargs)
2023-10-11T09:34:21.454588059Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 306, in _init_engine
2023-10-11T09:34:21.454671107Z     return engine_class(*args, **kwargs)
2023-10-11T09:34:21.454674152Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py", line 108, in __init__
2023-10-11T09:34:21.454676577Z     self._init_workers(distributed_init_method)
2023-10-11T09:34:21.454678901Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py", line 140, in _init_workers
2023-10-11T09:34:21.454689031Z     self._run_workers(
2023-10-11T09:34:21.454692468Z   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py", line 692, in _run_workers
2023-10-11T09:34:21.454895794Z     output = executor(*args, **kwargs)
2023-10-11T09:34:21.454916454Z   File "/usr/local/lib/python3.10/dist-packages/vllm/worker/worker.py", line 68, in init_model
2023-10-11T09:34:21.454920361Z     self.model = get_model(self.model_config)
2023-10-11T09:34:21.454923146Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/model_loader.py", line 101, in get_model
2023-10-11T09:34:21.454926492Z     model.load_weights(model_config.model, model_config.download_dir,
2023-10-11T09:34:21.454929719Z   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/mistral.py", line 392, in load_weights
2023-10-11T09:34:21.454950729Z     param = state_dict[name]
2023-10-11T09:34:21.454954215Z KeyError: 'model.layers.0.mlp.down_proj.qweight'
```






mistralai/Mistral-7B-v0.1 (_without_ GPTQ-optimized vLLM), ``:
```bash
sudo docker build  . -t matthewmcateer0/worker-runpod-vllm:Mistral7B-vllm0.2.0 --progress=plain --no-cache --platform linux/amd64  --build-arg HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXX --build-arg HF_MODEL_QUANTIZE=none --build-arg MODEL_NAME=mistralai/Mistral-7B-v0.1 --build-arg TOKENIZER=hf-internal-testing/llama-tokenizer --build-arg STREAMING=False --build-arg DOWNLOAD_MODEL=1

sudo docker push matthewmcateer0/worker-runpod-vllm:Mistral7B-vllm0.2.0
```



Please make sure to replace your_hugging_face_token_here with your actual Hugging Face token to enable model downloads that require it.

Ensure that you have Docker installed and properly set up before running the docker build commands. Once built, you can deploy this serverless worker in your desired environment with confidence that it will automatically scale based on demand. For further inquiries or assistance, feel free to contact our support team.


## Model Inputs
```
| Argument           | Type            | Default   | Description                                                                                                                                                      |
|--------------------|-----------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| n                  | int             | 1         | Number of output sequences to return for the given prompt.                                                                                                      |
| best_of            | Optional[int]   | None      | Number of output sequences that are generated from the prompt. From these `best_of` sequences, the top `n` sequences are returned. `best_of` must be greater than or equal to `n`. This is treated as the beam width when `use_beam_search` is True. By default, `best_of` is set to `n`. |
| presence_penalty   | float           | 0.0       | Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.                        |
| frequency_penalty  | float           | 0.0       | Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.                          |
| temperature        | float           | 1.0       | Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.                                        |
| top_p              | float           | 1.0       | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.                            |
| top_k              | int             | -1        | Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.                                                               |
| use_beam_search    | bool            | False     | Whether to use beam search instead of sampling.                                                                                                             |
| stop               | Union[None, str, List[str]] | None | List of strings that stop the generation when they are generated. The returned output will not contain the stop strings.                       |
| ignore_eos         | bool            | False     | Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.                                                            |
| max_tokens         | int             | 256       | Maximum number of tokens to generate per output sequence.                                                                                                   |
| logprobs           | Optional[int]   | None      | Number of log probabilities to return per output token.                                                                                                     |
```

## Test Inputs
The following inputs can be used for testing the model:
```json
{
    "input": {
       "prompt": "Who is the president of the United States?",
       "sampling_params": {
           "max_tokens": 100
       }
    }
}
```
