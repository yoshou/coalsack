#!/usr/bin/env bash
# deploy.sh — Deploy graph_proc_server to Raspberry Pi(s) via SSH/SCP
#
# Usage:
#   ./docker/deploy.sh [--dry-run] <user>@<ip> [<user>@<ip> ...]
#
# Examples:
#   ./docker/deploy.sh pi@192.168.1.10
#   ./docker/deploy.sh pi@192.168.1.10 ubuntu@192.168.1.11
#   ./docker/deploy.sh --dry-run pi@192.168.1.10
#
# Prerequisites:
#   - docker/output/image-armv7.tar and/or image-arm64.tar must exist (run build.sh first)
#   - SSH access to each Pi (password authentication supported)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
IMAGE_NAME="coalsack/graph_proc_server"
CONTAINER_NAME="graph_proc_server"
REMOTE_TMP="/tmp"
DRY_RUN=false
TARGETS=()
SSH_CONTROL_DIR="$(mktemp -d)"
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="${SSH_CONTROL_DIR}/%r@%h:%p" -o ControlPersist=60s)

# ------------------------------------------------------------------ #
usage() {
    echo "Usage: $0 [--dry-run] <user>@<ip> [<user>@<ip> ...]"
    exit 1
}

log()     { echo "[deploy.sh] $*"; }
log_dry() { echo "[deploy.sh][dry-run] $*"; }

# Run a command, or print it if --dry-run
run() {
    if [[ "${DRY_RUN}" == true ]]; then
        log_dry "$*"
    else
        "$@"
    fi
}

ssh_run() {
    local target="$1"
    shift
    if [[ "${DRY_RUN}" == true ]]; then
        log_dry "ssh ${target} -- $*"
    else
        ssh "${SSH_OPTS[@]}" "${target}" -- "$@"
    fi
}

scp_send() {
    local src="$1"
    local dst="$2"
    if [[ "${DRY_RUN}" == true ]]; then
        log_dry "scp ${src} ${dst}"
    else
        scp -o ControlMaster=auto -o ControlPath="${SSH_CONTROL_DIR}/%r@%h:%p" -o ControlPersist=60s "${src}" "${dst}"
    fi
}

open_master() {
    local target="$1"
    if [[ "${DRY_RUN}" != true ]]; then
        log "Opening SSH master connection to ${target}..."
        ssh "${SSH_OPTS[@]}" -o ControlMaster=yes "${target}" true
    fi
}

close_master() {
    local target="$1"
    if [[ "${DRY_RUN}" != true ]]; then
        ssh -O exit -o ControlPath="${SSH_CONTROL_DIR}/%r@%h:%p" "${target}" 2>/dev/null || true
    fi
}

# ------------------------------------------------------------------ #
# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *@*)
            TARGETS+=("$1")
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "Error: no target specified."
    usage
fi

# ------------------------------------------------------------------ #
# Detect git hash for tagging
GIT_HASH="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")"

# ------------------------------------------------------------------ #
deploy_target() {
    local target="$1"
    log "=========================================="
    log "Target: ${target}"
    log "=========================================="

    # Open a single master SSH connection (1 password prompt)
    open_master "${target}"

    # 1. Detect architecture
    log "Detecting architecture..."
    if [[ "${DRY_RUN}" == true ]]; then
        log_dry "ssh ${target} -- uname -m"
        local arch="armv7l"   # assume for dry-run display
        log_dry "(assuming armv7l for dry-run)"
    else
        local arch
        arch="$(ssh "${SSH_OPTS[@]}" "${target}" -- uname -m)"
    fi
    log "Architecture: ${arch}"

    # Map to arch tag and tar file
    local arch_tag tar_file full_tag
    case "${arch}" in
        armv7l|armv6l)
            arch_tag="armv7"
            ;;
        aarch64)
            arch_tag="arm64"
            ;;
        *)
            echo "Error: unsupported architecture '${arch}' on ${target}"
            return 1
            ;;
    esac
    tar_file="${OUTPUT_DIR}/image-${arch_tag}.tar"
    full_tag="${IMAGE_NAME}:${GIT_HASH}-${arch_tag}"

    # 2. Check tar file exists
    if [[ ! -f "${tar_file}" ]]; then
        echo "Error: ${tar_file} not found."
        echo "  Run: ./docker/build.sh --platform linux/${arch_tag/armv7/arm\/v7}"
        return 1
    fi
    log "Image tar: ${tar_file} ($(du -sh "${tar_file}" | cut -f1))"

    # 3. Install Docker if not present
    log "Checking Docker installation..."
    if [[ "${DRY_RUN}" == true ]]; then
        log_dry "ssh ${target} -- docker --version"
        log_dry "ssh ${target} -- curl -fsSL https://get.docker.com | sudo sh"
        log_dry "ssh ${target} -- sudo usermod -aG docker \$USER"
        log_dry "ssh ${target} -- sudo systemctl enable --now docker"
    else
        if ! ssh "${SSH_OPTS[@]}" "${target}" -- docker --version &>/dev/null; then
            log "Docker not found. Installing..."
            ssh "${SSH_OPTS[@]}" "${target}" -- 'curl -fsSL https://get.docker.com | sudo sh'
            ssh "${SSH_OPTS[@]}" "${target}" -- 'sudo usermod -aG docker $USER'
            ssh "${SSH_OPTS[@]}" "${target}" -- 'sudo systemctl enable --now docker'
            log "Docker installed and enabled."
        else
            log "Docker already installed: $(ssh "${SSH_OPTS[@]}" "${target}" -- docker --version)"
        fi
    fi

    # 4. Transfer image tar
    log "Transferring image to ${target}:${REMOTE_TMP}/..."
    scp_send "${tar_file}" "${target}:${REMOTE_TMP}/image-${arch_tag}.tar"

    # 5. Load image
    log "Loading image on remote..."
    ssh_run "${target}" "docker load -i ${REMOTE_TMP}/image-${arch_tag}.tar"

    # 6. Stop and remove existing container
    log "Stopping existing container (if any)..."
    ssh_run "${target}" "docker stop ${CONTAINER_NAME} 2>/dev/null || true"
    ssh_run "${target}" "docker rm   ${CONTAINER_NAME} 2>/dev/null || true"

    # 7. Start new container
    log "Starting container..."
    ssh_run "${target}" \
        "docker run -d \
            --name ${CONTAINER_NAME} \
            --restart unless-stopped \
            --network host \
            --privileged \
            ${full_tag}"

    # 8. Cleanup remote tar and old images
    log "Cleaning up remote tar and dangling images..."
    ssh_run "${target}" "rm -f ${REMOTE_TMP}/image-${arch_tag}.tar"
    ssh_run "${target}" "docker image prune -f"

    # 9. Verify
    log "Container status:"
    ssh_run "${target}" "docker ps --filter name=${CONTAINER_NAME} --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"

    log "Deploy to ${target} complete."
}

# ------------------------------------------------------------------ #
for target in "${TARGETS[@]}"; do
    deploy_target "${target}"
    close_master "${target}"
done

rm -rf "${SSH_CONTROL_DIR}"

log ""
log "All deployments complete."
