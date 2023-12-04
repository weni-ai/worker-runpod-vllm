import os
from huggingface_hub import snapshot_download

# Get the hugging face token
HUGGING_FACE_HUB_TOKEN = os.environ.get('HUGGING_FACE_HUB_TOKEN', None)
MODEL_NAME = os.environ.get('MODEL_NAME')
MODEL_REVISION = os.environ.get('MODEL_REVISION', "main")

# Download the model from hugging face
download_kwargs = {}

if HUGGING_FACE_HUB_TOKEN:
    download_kwargs["token"] = HUGGING_FACE_HUB_TOKEN

snapshot_download(
    repo_id=MODEL_NAME,
    revision=MODEL_REVISION,
    **download_kwargs
)
