FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install needed packages. Use a separate RUN statement to add your own dependencies.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install build-essential cmake cppcheck valgrind clang lldb llvm gdb \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install sudo locales \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Setup ENV vars for vcpkg
ENV VCPKG_ROOT=/usr/local/vcpkg \
    VCPKG_DOWNLOADS=/usr/local/vcpkg-downloads
ENV PATH="${PATH}:${VCPKG_ROOT}"

RUN locale-gen en_US.UTF-8  
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

ARG USERNAME=vscode
ARG GROUPNAME=vscode
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME

# Install vcpkg itself: https://github.com/microsoft/vcpkg/blob/master/README.md#quick-start-unix
COPY ./install-vcpkg.sh /tmp/
RUN chmod +x /tmp/install-vcpkg.sh && /tmp/install-vcpkg.sh ${USERNAME} \
    && rm -f /tmp/install-vcpkg.sh

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.22.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends gfortran bison python3 python3-pip python3-setuptools libarchive-dev libblas-dev liblapack-dev

RUN su vscode -c "${VCPKG_ROOT}/vcpkg install boost-asio opencv4[core,jpeg]"

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends autoconf automake libtool libudev-dev
    
RUN su vscode -c "${VCPKG_ROOT}/vcpkg install realsense2"

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install wget \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

WORKDIR /onnxruntime/

RUN curl -L \
    https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-gpu-1.15.1.tgz \
    -o /tmp/onnxruntime-linux-x64-gpu-1.15.1.tgz

RUN tar -zxvf /tmp/onnxruntime-linux-x64-gpu-1.15.1.tgz --strip-components 1 -C /onnxruntime

WORKDIR /tensorflow/

RUN curl -L \
    https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.10.1.tar.gz \
    -o /tmp/tensorflow-2.10.1.tar.gz \
    && tar -zxvf /tmp/tensorflow-2.10.1.tar.gz --strip-components 1 -C /tensorflow

ENV PATH $PATH:${VCPKG_ROOT}/installed/x64-linux
ENV BOOST_ROOT /usr/local/vcpkg/installed/x64-linux
ENV ONNXRUNTIME_ROOT /onnxruntime
ENV TENSORFLOW_ROOT /tensorflow

USER ${USERNAME}
