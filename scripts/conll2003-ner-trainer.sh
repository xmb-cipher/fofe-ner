#!/bin/bash

export THIS_DIR=$(cd $(dirname $0); pwd)
source ${THIS_DIR}/../path.sh

function cleanup() {
    echo ${2} | egrep "^/tmp"
    [ $? -eq 0 ] && rm -rf $2 &> /dev/null
}
trap "cleanup" EXIT

INFO "$@"
${THIS_DIR}/../conll2003-ner-trainer.py $@ |& tee ${THIS_DIR}/../nimei-`hostname`

exit $?
