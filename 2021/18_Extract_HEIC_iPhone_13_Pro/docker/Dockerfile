FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    autoconf \
    libtool \
    git \
    vim

RUN apt-get install doxygen -y

# RUN mkdir -p /work/src \
#     && cd /work/src \
#     && git clone https://github.com/yohhoy/heic2hevc.git \
#     && cd heic2hevc \
#     && rm -rf nokiatech-heif \
#     && git clone https://github.com/nokiatech/heif.git nokiatech-heif -b v3.7.0 --depth 1 
#     && cd nokiatech-heif/build \
#     && cmake ../srcs -G"Unix Makefiles" \
#     && cmake --build .

# RUN mkdir -p /root/local/src \
#     && cd /root/local/src \
#     && git clone https://github.com/yohhoy/heic2hevc.git \
#     && make

# RUN cd /root/local/src \
#     && tar -xzf IccXML-0.9.8.tar.gz \
#     && cd IccXML-0.9.8 \
#     && ./configure \
#     && make -j8 \
#     && make install

# RUN echo "/root/local/src/heif/build/lib" >> /etc/ld.so.conf && ldconfig

# RUN rm -rf /root/local/src \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
