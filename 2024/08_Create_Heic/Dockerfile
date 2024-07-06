FROM python:3.10.14-slim

RUN apt-get update && apt-get -y install --no-install-recommends \
    build-essential \
    cmake \
    git

RUN cd /root \
    && git clone https://github.com/nokiatech/heif.git -b v3.7.0 \
    && cd /root/heif/build \
    && sed -i '1s/^/#include <limits>\n/' ../srcs/reader/heifreaderimpl.cpp \
    && cmake ../srcs -G"Unix Makefiles" \
    && cmake --build .
