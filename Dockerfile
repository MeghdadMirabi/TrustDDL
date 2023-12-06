FROM ubuntu:22.04

RUN apt-get update\
    && apt-get install -y \
        nano \
        python3 \
        python3-pip

RUN mkdir TrustDDL

ADD TrustDDL TrustDDL

RUN pip install -r TrustDDL/requirements.txt