import torch
import gc
import time
import datetime
import whisperx
import base64
from potassium import Potassium, Request, Response
import os

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context


@app.init
def init():

    hf_auth_token = os.environ.get('HF_AUTH_TOKEN')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    model = whisperx.load_model(
        "large-v2", device=device, compute_type=compute_type)
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_auth_token, device=device)

    context = {
        "model": model,
        "diarize_model": diarize_model,
        "device": device,
    }

    return context


@app.handler()
def handler(context: dict, request: Request) -> Response:
    batch_size = 16

    base64file = request.json.get("file")
    number_speakers = request.json.get("number_speakers")

    if base64file == None or base64file == '':
        return {'message': "No correct input provided"}

    try:
        number_speakers = int(number_speakers)
    except ValueError:
        return {'message': "number_speakers not an integer"}

    model = context.get("model")
    diarize_model = context.get("diarize_model")
    device = context.get("device")

    base64_data = base64file.split(",")[1]
    file_data = base64.b64decode(base64_data)

    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    audio_file = f'{ts}'
    with open(audio_file, 'wb') as f:
        f.write(file_data)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    gc.collect()
    torch.cuda.empty_cache()
    del model

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device)
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    diarize_segments = diarize_model(
        audio_file, min_speakers=number_speakers, max_speakers=number_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    return Response(
        json={"output": result["segments"]},
        status=200
    )


if __name__ == "__main__":
    app.serve()
