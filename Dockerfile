FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y ffmpeg
RUN apt-get install -y --no-install-recommends \
	software-properties-common \
    libglib2.0-0 \
    libsm6 \ 
    libxext6 \ 
    libxrender-dev \
    python3-dev \
    libgl1-mesa-glx

RUN pip install numpy
RUN pip install Pillow
RUN pip install scikit-image
RUN pip install opencv-contrib-python-headless==3.4.7.28
RUN pip install torch==1.4.0
RUN pip install torchvision
RUN pip install tensorboard

WORKDIR /root/workspace
