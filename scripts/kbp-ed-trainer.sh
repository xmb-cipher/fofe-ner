#!/bin/bash

export THIS_DIR=$(cd $(dirname $0); pwd)
source ${THIS_DIR}/../path.sh

worker_id=$1
shift
INFO "worker-id == ${worker_id}"

export SHOW_ID=$(nvidia-smi -q | grep -i processes | grep -in none | cut -d':' -f1 | tail -1)

# TODO: this line should be changed according to GPU order
# e.g. GPU on image is in reversed order
export CUDA_VISIBLE_DEVICES=$((4 - ${SHOW_ID}))

function cleanup() {
    echo ${2} | egrep "^/tmp"
    [ $? -eq 0 ] && rm -rf $2 &> /dev/null
}
trap "cleanup" EXIT

INFO "$@"
${THIS_DIR}/../kbp-ed-trainer.py $@ #&> /dev/null

exit $?
