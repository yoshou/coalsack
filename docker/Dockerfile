ARG BASE_IMAGE=debian:bookworm

#################################
#   Librealsense Builder Stage  #
#################################
FROM $BASE_IMAGE as librealsense-builder

ARG LIBRS_VERSION=2.55.1
# Make sure that we have a version number of librealsense as argument
RUN test -n "$LIBRS_VERSION"

# To avoid waiting for input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Builder dependencies installation
RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libusb-1.0-0-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \    
    curl \
    python3 \
    python3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download sources
WORKDIR /usr/src
RUN curl https://codeload.github.com/IntelRealSense/librealsense/tar.gz/refs/tags/v$LIBRS_VERSION -o librealsense.tar.gz 
RUN tar -zxf librealsense.tar.gz \
    && rm librealsense.tar.gz 
RUN ln -s /usr/src/librealsense-$LIBRS_VERSION /usr/src/librealsense

# Build and install
RUN cd /usr/src/librealsense \
    && mkdir build && cd build \
    && cmake \
    -DCMAKE_C_FLAGS_RELEASE="${CMAKE_C_FLAGS_RELEASE} -s" \
    -DCMAKE_CXX_FLAGS_RELEASE="${CMAKE_CXX_FLAGS_RELEASE} -s" \
    -DCMAKE_INSTALL_PREFIX=/opt/librealsense \    
    -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_PYTHON_BINDINGS:bool=false \
    -DCMAKE_BUILD_TYPE=Release ../ \
    && make -j$(($(nproc)-1)) all \
    && make install 

##############################
#   Coalsack Builder Stage   #
##############################
FROM $BASE_IMAGE

# Copy binaries from builder stage
COPY --from=librealsense-builder /opt/librealsense /usr/local/
COPY --from=librealsense-builder /usr/src/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
COPY --from=librealsense-builder /usr/src/librealsense/config/99-realsense-d4xx-mipi-dfu.rules /etc/udev/rules.d/
ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib

# Install dep packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libusb-1.0-0 \
    udev \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y gcc g++ git cmake automake libboost-all-dev libopencv-dev libyaml-dev python3-yaml python3-ply \
                       python3-jinja2 python3-pip libgnutls28-dev && \
    apt-get -y --purge autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y ninja-build && \
    apt-get -y --purge autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip --break-system-packages && pip3 install meson --break-system-packages

ENV CC=/usr/bin/gcc \
    CXX=/usr/bin/g++

RUN git clone https://github.com/raspberrypi/libcamera.git && \
    cd libcamera && \
    meson build && \
    ninja -C build install

RUN git clone https://github.com/RPi-Distro/raspi-gpio.git && \
    cd raspi-gpio && \
    autoreconf && \
    ./configure && \
    make && \
    make install

WORKDIR /app

RUN git clone https://github.com/yoshou/coalsack /app/coalsack

WORKDIR /app/coalsack/build

RUN cd /app/coalsack/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j

ENV LD_LIBRARY_PATH=/usr/local/lib/arm-linux-gnueabihf:$LD_LIBRARY_PATH

CMD ["/app/coalsack/build/graph_proc_server"]
