FROM python:3.10.14-slim

WORKDIR /root/local

RUN apt-get update && apt-get -y install --no-install-recommends \
    build-essential \
    git \
    ninja-build \
    cmake \
    libjpeg-dev \
    clang

RUN git clone https://github.com/google/libultrahdr.git \
    && cd /root/local/libultrahdr \
    && mkdir build_directory \
    && cd build_directory \
    && cmake -G "Unix Makefiles" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DUHDR_BUILD_TESTS=1 ../ \
    && make \
    && cp ultrahdr_app /usr/local/bin/ \
    && cd / \
    && rm -rf /root/local/libultrahdr

# Clean
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /roor/local/src

# ENTRYPOINT ["/opt/ultrahdr/ultrahdr_app"]
CMD ["bash"]
