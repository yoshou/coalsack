ARG BASE_IMAGE=debian:bookworm

##########################
#   Builder Stage        #
##########################
FROM $BASE_IMAGE AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and -dev packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential gcc g++ git cmake automake \
    ninja-build \
    libboost-all-dev \
    libopencv-dev \
    nlohmann-json3-dev \
    libyaml-dev python3-yaml python3-ply python3-jinja2 python3-pip \
    libgnutls28-dev \
    pkg-config \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip --break-system-packages && \
    pip3 install meson --break-system-packages

ENV CC=/usr/bin/gcc \
    CXX=/usr/bin/g++

# Build and install libcamera
RUN git clone https://github.com/raspberrypi/libcamera.git /tmp/libcamera && \
    cd /tmp/libcamera && \
    meson build --prefix=/usr/local && \
    ninja -C build install && \
    rm -rf /tmp/libcamera

# Build and install raspi-gpio
RUN git clone https://github.com/RPi-Distro/raspi-gpio.git /tmp/raspi-gpio && \
    cd /tmp/raspi-gpio && \
    autoreconf && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    rm -rf /tmp/raspi-gpio

# Copy coalsack source from local workspace
COPY . /app/coalsack

WORKDIR /app/coalsack/build

RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_LIBCAMERA_EXT=ON \
    -DENABLE_LLM_OPS=OFF \
    && make -j$(nproc) graph_proc_server

##########################
#   Runtime Stage        #
##########################
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime libraries only (no -dev packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libopencv-core4.6 \
    libopencv-imgproc4.6 \
    libopencv-imgcodecs4.6 \
    libopencv-calib3d4.6 \
    libopencv-features2d4.6 \
    libopencv-video4.6 \
    libopencv-highgui4.6 \
    libopencv-videoio4.6 \
    libopencv-contrib406 \
    libboost-system1.74.0 \
    libboost-filesystem1.74.0 \
    libboost-thread1.74.0 \
    libboost-atomic1.74.0 \
    libboost-date-time1.74.0 \
    libboost-regex1.74.0 \
    libboost-chrono1.74.0 \
    libboost-log1.74.0 \
    libyaml-0-2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy libcamera and raspi-gpio runtime libraries from builder
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/share/libcamera /usr/local/share/libcamera
COPY --from=builder /usr/local/bin/raspi-gpio /usr/local/bin/raspi-gpio

# Copy the server binary
COPY --from=builder /app/coalsack/build/graph_proc_server /app/graph_proc_server

# Configure dynamic linker to find libraries in /usr/local/lib and multiarch paths
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
        aarch64) MULTIARCH="aarch64-linux-gnu" ;; \
        armv7l)  MULTIARCH="arm-linux-gnueabihf" ;; \
        *)       MULTIARCH="$ARCH-linux-gnu" ;; \
    esac && \
    echo "/usr/local/lib/${MULTIARCH}" >> /etc/ld.so.conf.d/local.conf && \
    ldconfig

EXPOSE 31400

CMD ["/app/graph_proc_server"]
