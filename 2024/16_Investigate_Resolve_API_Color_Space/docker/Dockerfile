FROM rockylinux:9.3

WORKDIR /root/local

RUN dnf install -y \
        mesa-libGL \
        mesa-libGLU \
        libxcb \
        libxkbcommon \
        xcb-util \
        xcb-util-wm \
        qt5-qtbase \
        qt5-qtbase-gui \
        librsvg2 \
        ocl-icd \
        libxcrypt-compat

COPY DaVinci_Resolve_Studio_19.1_Linux.run .
RUN chmod 755 DaVinci_Resolve_Studio_19.1_Linux.run . \
    && ./DaVinci_Resolve_Studio_19.1_Linux.run --appimage-extract \
    && cd ./squashfs-root/ \
    && ./installer -i -a -y .

CMD ["bash"]
