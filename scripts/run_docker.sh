#!/usr/bin/env bash

# To enable gpus, add: --gpus all \
# To enable visualizer port, add: -p 8888:8888 \

# --mount type=bind,source=/zfsauton2/home/gwelter/code/cmu-ta2-vi,target=/vis \

docker run -it \
    --rm \
    --name d3m \
    --gpus all \
    --mount type=bind,source=/home/gwelter/datasets-new-20200713,target=/data,readonly \
    --mount type=bind,source=/zfsauton2/home/gwelter/code/cmu-ta2,target=/ta2 \
    --mount type=bind,source=/zfsauton2/home/gwelter/code/ta2vi-orig,target=/vis \
    --mount type=bind,source=/zfsauton/data/public/gwelter/ta2jobs,target=/jobs \
    registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.5.18-20200630-050709 \
    /ta2/scripts/test.sh #/bin/bash #/ta2/scripts/test.sh #/vis/scripts/start.sh
