#!/usr/bin/env bash
# docker-run-label.sh
# Launches the label_person Docker container with X11 display forwarding.
#
# Usage:
#   ./docker-run-label.sh --images_dir /absolute/path/to/frames
#   ./docker-run-label.sh --images_dir /abs/images --labels_dir /abs/labels
#
# All arguments are forwarded to label_person.py.

set -euo pipefail

IMAGE="label-person"

# Allow the container to connect to the host X server
xhost +local:docker 2>/dev/null || true

# Resolve the host directory to mount from --images_dir (first occurrence)
HOST_DATA=""
for arg in "$@"; do
    if [[ -n "$HOST_DATA" ]]; then
        HOST_DATA="$arg"
        break
    fi
    [[ "$arg" == "--images_dir" ]] && HOST_DATA="next"
done

if [[ -z "$HOST_DATA" || "$HOST_DATA" == "next" ]]; then
    echo "Usage: $0 --images_dir /absolute/path [--labels_dir /absolute/path] [...]"
    exit 1
fi

# Convert CLI paths to container paths (/data/<basename>) for images_dir and labels_dir
CONTAINER_ARGS=()
MOUNTS=()
PREV=""
for arg in "$@"; do
    if [[ "$PREV" == "--images_dir" || "$PREV" == "--labels_dir" ]]; then
        BASENAME=$(basename "$arg")
        MOUNTS+=("-v" "${arg}:/data/${BASENAME}:rw")
        CONTAINER_ARGS+=("/data/${BASENAME}")
    else
        CONTAINER_ARGS+=("$arg")
    fi
    PREV="$arg"
done

docker run --rm -it \
    --network host \
    -e DISPLAY="${DISPLAY}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    "${MOUNTS[@]}" \
    "$IMAGE" \
    "${CONTAINER_ARGS[@]}"
