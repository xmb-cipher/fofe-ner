#!/bin/bash

export THIS_DIR=$(cd $(dirname $0); pwd)
source ${THIS_DIR}/../path.sh

INFO "$@"
${THIS_DIR}/../kbp-ed-trainer.py $@
