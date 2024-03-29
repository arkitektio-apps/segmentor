FROM tensorflow/tensorflow:latest-gpu


RUN pip install stardist gputools edt 
RUN pip install "arkitekt[all]==0.5.57"
RUN pip install "pydantic<2"

RUN mkdir /workspace
COPY . /workspace
WORKDIR /workspace
