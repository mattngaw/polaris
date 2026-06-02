#!/usr/bin/env bash
# Host-side wrapper for the polaris ROCm container.
#
#   ./env/run-container.sh           # start (or attach to) the container;
#                                    # drops you in a /workspace shell.
#   ./env/run-container.sh shell     # same as default.
#   ./env/run-container.sh status    # report state of the named container.
#   ./env/run-container.sh stop      # stop and remove the container.
#
# The container is persistent (no --rm). Run this script a second time from
# another host shell to get a second shell into the same container — that
# is the two-card pattern (HIP_VISIBLE_DEVICES=0 in one, =1 in the other).
#
# Configurable via env:
#   POLARIS_IMAGE       (default: robertrosenbusch/rocm6_gfx803_comfyui:5.7)
#   POLARIS_CONTAINER   (default: polaris-dev)
#   POLARIS_HOST_DIR    (default: this script's parent dir, resolved)
#   POLARIS_MOUNT       (default: /workspace)
#   DOCKER              (default: docker; set to "sudo docker" if your user
#                        is not in the `docker` group)
#
# The 5.7 image's default entrypoint launches ComfyUI on port 8188; we
# override with --entrypoint bash so the container is a plain shell.
# Conda env is at /opt/conda/envs/py_3.10 — `python` is on PATH inside.
# Why 5.7: ROCm 6.x has a gfx803 fp32 GEMM correctness bug (see
# KNOWN_GOTCHAS.md). 5.7 is the community-validated path where fp32
# works.

set -euo pipefail

IMAGE="${POLARIS_IMAGE:-robertrosenbusch/rocm6_gfx803_comfyui:5.7}"
NAME="${POLARIS_CONTAINER:-polaris-dev}"
HOST_DIR="${POLARIS_HOST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MOUNT="${POLARIS_MOUNT:-/workspace}"
DOCKER="${DOCKER:-docker}"

container_running() {
  $DOCKER ps --format '{{.Names}}' | grep -qx "$NAME"
}

container_exists() {
  $DOCKER ps -a --format '{{.Names}}' | grep -qx "$NAME"
}

cmd="${1:-shell}"

case "$cmd" in
  shell)
    if container_running; then
      exec $DOCKER exec -it "$NAME" bash
    elif container_exists; then
      $DOCKER start "$NAME" >/dev/null
      exec $DOCKER exec -it "$NAME" bash
    else
      exec $DOCKER run -it \
        --name "$NAME" \
        --privileged \
        --network host \
        --ipc host \
        --shm-size 16G \
        --group-add video \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --device /dev/kfd \
        --device /dev/dri \
        -e HSA_OVERRIDE_GFX_VERSION=8.0.3 \
        -e PYTORCH_ROCM_ARCH=gfx803 \
        -e PYTHONUNBUFFERED=1 \
        -v "$HOST_DIR:$MOUNT" \
        -w "$MOUNT" \
        --entrypoint bash \
        "$IMAGE"
    fi
    ;;
  status)
    if container_running; then
      echo "$NAME: running"
    elif container_exists; then
      echo "$NAME: stopped"
    else
      echo "$NAME: not created"
    fi
    ;;
  stop)
    if container_running; then
      $DOCKER stop "$NAME" >/dev/null
    fi
    if container_exists; then
      $DOCKER rm "$NAME" >/dev/null
    fi
    echo "$NAME: stopped and removed"
    ;;
  *)
    echo "usage: $0 [shell|status|stop]" >&2
    exit 2
    ;;
esac
