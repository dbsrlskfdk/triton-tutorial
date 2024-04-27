#!/bin/bash

HERE=$(dirname "$(readlink -f $0)")
PARENT_DIR=$(dirname "$HERE")

docker run -it --rm --name trtis \
    -p8100:8000   \
    -p8101:8001   \
    -p8102:8002    \
    -v "${PARENT_DIR}"/model_repository:/models:ro \
    -e OMP_NUM_THREADS=2 \
    -e OPENBLAS_NUM_THREADS=2 \
    --shm-size=1g  \
    --gpus all \
    triton-vad-server \
    tritonserver --model-repository=/models \
    --exit-timeout-secs 15 \
    --min-supported-compute-capability 7.0 \
