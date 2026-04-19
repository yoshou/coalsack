#!/usr/bin/env bash
# build.sh — Build graph_proc_server Docker images for Raspberry Pi (armv7 / arm64)
#
# Usage:
#   ./docker/build.sh --platform linux/arm/v7
#   ./docker/build.sh --platform linux/arm64
#   ./docker/build.sh --platform all
#
# Output:
#   docker/output/image-armv7.tar
#   docker/output/image-arm64.tar

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"
IMAGE_NAME="coalsack/graph_proc_server"
BUILDER_NAME="pibuilder"

PLATFORM="all"

usage() {
    echo "Usage: $0 --platform <linux/arm/v7 | linux/arm64 | all>"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Resolve git tag for image tagging
GIT_HASH="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")"

mkdir -p "${OUTPUT_DIR}"

# Setup QEMU binfmt handlers for ARM cross-build
echo "[build.sh] Setting up QEMU binfmt handlers..."
docker run --privileged --rm tonistiigi/binfmt --install all

# Ensure buildx builder exists
if ! docker buildx inspect "${BUILDER_NAME}" &>/dev/null; then
    echo "[build.sh] Creating buildx builder '${BUILDER_NAME}'..."
    docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use
else
    docker buildx use "${BUILDER_NAME}"
fi
docker buildx inspect --bootstrap > /dev/null

build_arch() {
    local platform="$1"   # e.g. linux/arm/v7
    local arch_tag="$2"   # e.g. armv7
    local tar_file="${OUTPUT_DIR}/image-${arch_tag}.tar"
    local full_tag="${IMAGE_NAME}:${GIT_HASH}-${arch_tag}"
    local latest_tag="${IMAGE_NAME}:latest-${arch_tag}"

    echo ""
    echo "================================================================"
    echo "[build.sh] Building for ${platform} → ${tar_file}"
    echo "  Tags: ${full_tag}, ${latest_tag}"
    echo "================================================================"

    docker buildx build \
        --platform "${platform}" \
        --file "${DOCKERFILE}" \
        --tag "${full_tag}" \
        --tag "${latest_tag}" \
        --load \
        "${REPO_ROOT}"

    echo "[build.sh] Saving image to ${tar_file}..."
    docker save "${full_tag}" -o "${tar_file}"
    echo "[build.sh] Done: ${tar_file} ($(du -sh "${tar_file}" | cut -f1))"
}

case "${PLATFORM}" in
    linux/arm/v7)
        build_arch "linux/arm/v7" "armv7"
        ;;
    linux/arm64)
        build_arch "linux/arm64" "arm64"
        ;;
    all)
        build_arch "linux/arm/v7" "armv7"
        build_arch "linux/arm64"  "arm64"
        ;;
    *)
        echo "Unknown platform: ${PLATFORM}"
        usage
        ;;
esac

echo ""
echo "[build.sh] All builds complete."
echo "  Output directory: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"/*.tar 2>/dev/null || true
