FROM tensorflow/tensorflow:latest-gpu


RUN pip install stardist gputools edt 
RUN pip install "arkitekt[cli]==0.4.113"


RUN mkdir /workspace
COPY . /workspace
WORKDIR /workspace

CMD python predicter.py