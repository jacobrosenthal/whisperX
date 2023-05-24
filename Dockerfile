# Banana says Must use a Cuda version 11+
# whisperx requires 3.8.1, assume this is that?
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ARG HF_AUTH_TOKEN
ENV HF_AUTH_TOKEN=$HF_AUTH_TOKEN

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y build-essential git
RUN apt-get install -y ffmpeg

# Install python packages
RUN pip3 install --upgrade pip
RUN pip3 install potassium
RUN pip3 install -U torch==2.0.0+cu117 torchvision==0.15.0 torchaudio==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu117
# not great for caching presumably
COPY . .
RUN pip3 install -e .

# Add your model weight files 
# (in this case we have a python script)
# ADD download.py .
RUN python3 download.py

# Add your custom app code, init() and inference()
# ADD app.py .

EXPOSE 8000

CMD python3 -u app.py