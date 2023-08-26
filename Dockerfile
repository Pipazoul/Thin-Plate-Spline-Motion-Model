FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y gcc cmake build-essential git wget

RUN git clone https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model.git

WORKDIR /app/Thin-Plate-Spline-Motion-Model

RUN mkdir checkpoints
RUN wget -c https://stableai-space.fra1.digitaloceanspaces.com/vox.pth.tar -O checkpoints/vox.pth.tar

RUN pip install fastapi uvicorn
RUN pip install imageio imageio-ffmpeg
RUN pip install matplotlib
RUN pip install scikit-image

COPY main.py /app/Thin-Plate-Spline-Motion-Model/main.py


RUN apt install ffmpeg -y

# reload and debug
ENTRYPOINT [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000" , "--reload"]