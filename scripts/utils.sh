#!/bin/bash

export LANG=C

# these 3 machines have pascal GPU installed
# the rest are TitanX
if [ `hostname` == "image" ] || [ `hostname` == "voice" ] || [ `hostname` == "audio" ]
then
    export CUDA_HOME="/eecs/local/pkg/cuda-8.0.44"
else
    export CUDA_HOME="/eecs/local/pkg/cuda"
fi

export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/eecs/research/asr/Shared/cuDNN/lib64:${LD_LIBRARY_PATH}


export KNRM="\x1B[0m"
export KRED="\x1B[31m"
export KGRN="\x1B[32m"
export KYEL="\x1B[33m"
export KBLU="\x1B[34m"
export KMAG="\x1B[35m"
export KCYN="\x1B[36m"
export KWHT="\x1B[37m"


function INFO() {
    msg="$@"
    printf "${KGRN}"
    printf "`date +"%Y-%m-%d %H-%M-%S"` [INFO]: ${msg}\n"
    printf "${KNRM}"
}


function CRITICAL() {
    msg="$@"
    printf "${KRED}"
    printf "`date +"%Y-%m-%d %H-%M-%S"` [CRITICAL]: ${msg}\n"
    printf "${KNRM}"
}