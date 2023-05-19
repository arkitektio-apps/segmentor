FROM tensorflow/tensorflow:latest-gpu


RUN pip install stardist gputools edt 
RUN pip install "arkitekt[cli]==0.4.111"

#RUN pip install grunnlag==0.4.5 s3fs==0.4.2 # 04.2 because its the last working s3fs for freeking python 3.6.9
#RUN pip install bergen==0.4.32

RUN mkdir /workspace
COPY . /workspace
WORKDIR /workspace

CMD python predicter.py