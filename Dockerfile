FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="ko.sugawara@ens-lyon.fr"

ARG NVIDIA_DRIVER_VERSION=515

RUN apt-get update && apt-get install -y --no-install-recommends \
    ocl-icd-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo \
    libnvidia-compute-${NVIDIA_DRIVER_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install stardist gputools edt arkitekt==0.4.13

#RUN pip install grunnlag==0.4.5 s3fs==0.4.2 # 04.2 because its the last working s3fs for freeking python 3.6.9
#RUN pip install bergen==0.4.32

RUN mkdir /workspace
COPY . /workspace
WORKDIR /workspace

CMD python predicter.py