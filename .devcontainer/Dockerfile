FROM mcr.microsoft.com/devcontainers/cpp:1-debian12

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.22.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# Install build dependencies and required libraries via apt
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    gfortran bison python3 python3-pip python3-setuptools \
    libarchive-dev libblas-dev liblapack-dev \
    libboost-all-dev \
    libopencv-dev \
    nlohmann-json3-dev \
    autoconf automake libtool libudev-dev \
    wget curl \
    usbutils udev \
    software-properties-common apt-transport-https \
    clang-format \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Build and install librealsense from source (v2.56.5)
# Following official documentation: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev \
    libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev v4l-utils \
    && cd /tmp \
    && wget https://github.com/realsenseai/librealsense/archive/refs/tags/v2.56.5.tar.gz \
    && tar -xzf v2.56.5.tar.gz \
    && cd librealsense-2.56.5 \
    && ./scripts/setup_udev_rules.sh || true \
    && mkdir build && cd build \
    && cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd /tmp \
    && rm -rf librealsense-2.56.5 v2.56.5.tar.gz \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

WORKDIR /onnxruntime/

# Install ONNX Runtime CPU version
RUN curl -L \
    https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-1.15.1.tgz \
    -o /tmp/onnxruntime-linux-x64-1.15.1.tgz \
    && tar -zxvf /tmp/onnxruntime-linux-x64-1.15.1.tgz --strip-components 1 -C /onnxruntime \
    && rm /tmp/onnxruntime-linux-x64-1.15.1.tgz

# Install DepthAI
RUN curl -L \
    https://github.com/luxonis/depthai-core/releases/download/v2.32.0/depthai-core-v2.32.0.tar.gz \
    -o /tmp/depthai-core-v2.32.0.tar.gz \
    && mkdir /tmp/depthai-core \
    && tar -zxvf /tmp/depthai-core-v2.32.0.tar.gz --strip-components 1 -C /tmp/depthai-core \
    && cd /tmp/depthai-core \
    && cmake -S. -Bbuild -DDEPTHAI_OPENCV_SUPPORT=ON -DDEPTHAI_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--exclude-libs,ALL" \
    && cmake --build build --target install \
    && rm -rf /tmp/depthai-core /tmp/depthai-core-v2.32.0.tar.gz

# Setup udev rules for USB devices
RUN echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | tee /etc/udev/rules.d/80-movidius.rules

# Install libcamera build dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    ninja-build libyaml-dev python3-yaml python3-ply python3-jinja2 libgnutls28-dev libdrm-dev \
    && pip3 install meson --break-system-packages \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Build and install libcamera from official repository
RUN git clone https://git.libcamera.org/libcamera/libcamera.git /tmp/libcamera \
    && cd /tmp/libcamera \
    && meson setup build --prefix=/usr/local \
    && ninja -C build install \
    && ldconfig \
    && rm -rf /tmp/libcamera

# Set environment variables
ENV ONNXRUNTIME_ROOT=/onnxruntime

USER vscode
