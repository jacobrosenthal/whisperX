# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import torch
import whisperx
import os


def download_model():
    hf_auth_token = os.environ.get('HF_AUTH_TOKEN')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    model = whisperx.load_model(
        "large-v2", device=device, compute_type=compute_type)
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_auth_token, device=device)


if __name__ == "__main__":
    download_model()
