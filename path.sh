#!/bin/bash

export LANG=C
export EXPT="/eecs/research/asr/mingbin/ner-advance"
export LOCAL_SCRIPT=${EXPT}/scripts

##############
# CUDA-related

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-""}

if [ `hostname` == "image" ] || [ `hostname` == "voice" ] || [ `hostname` == "audio" ]
then
    export CUDA_HOME="/eecs/local/pkg/cuda-8.0.44"
else
    export CUDA_HOME="/eecs/local/pkg/cuda"
fi

export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# export LD_LIBRARY_PATH=/eecs/research/asr/Shared/cuDNN/lib64:${LD_LIBRARY_PATH}


#############
# GCC-related

export PATH=/eecs/research/asr/mingbin/gcc-4.9/bin:${PATH}
export LIBRARY_PATH=/eecs/research/asr/mingbin/gcc-4.9/lib64
export LD_LIBRARY_PATH=/eecs/research/asr/mingbin/gcc-4.9/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/eecs/research/asr/mingbin/mkl/mkl/lib/intel64:${LD_LIBRARY_PATH}
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32 


################
# PYTHON-related

source /eecs/research/asr/mingbin/python-workspace/hopeless/bin/activate
export PYTHONPATH=${EXPT}:${LOCAL_SCRIPT}
export NLTK_DATA=/eecs/research/asr/mingbin/nltk-data


###############
# debug-related

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


function ServerList() {
    wanted=${1}
    wanted=${wanted:-75}
    cnt=0

    for i in `seq 1 75`
    do 
        h=ea`printf "%02d" ${i}`
        ping -c 1 ${h} > /dev/null
        if [ $? -eq 0 ] 
        then
            buff=`ssh ${h} "ps aux | cut -d' ' -f1 | grep xmb | wc -l; head -1 /proc/meminfo | tr -s ' '  | cut -d' ' -f2"`
            n_user=`echo ${buff} | tail -1 | cut -d' ' -f1`
            k_mem=`echo ${buff} | tail -1 | cut -d' ' -f2`
            if [ ${n_user} -le 8 ] && [ ${k_mem} -gt 16000000 ]
            then 
                echo ${h}
                cnt=$((cnt + 1))
                [ ${cnt} -eq ${wanted} ] && break
            fi
        fi
    done
}



function RUN1 {
    export VERSION=1
    for lang in spa cmn eng
    do
        for yr in 2016 2015
        do
            export KBP_NFOLD_LANG=${lang}
            export YEAR=${yr}
            ${EXPT}/kbp-nfold-trainer.sh \
                |& tee ${EXPT}/kbp-result/${lang}-${yr}-v${VERSION}.log
            sleep 60
        done
    done
}



function RUN2 {
    export VERSION=2
    for lang in spa cmn eng
    do
        for yr in 2016 2015
        do
            export KBP_NFOLD_LANG=${lang}
            export YEAR=${yr}
            ${EXPT}/kbp-nfold-trainer.sh \
                |& tee ${EXPT}/kbp-result/${lang}-${yr}-v${VERSION}.log
            sleep 60
        done
    done
}


function RunV2 {
    export VERSION=2
    lang=${1}
    lang=${lang:-eng}
    for yr in 2016 2015
    do
        export KBP_NFOLD_LANG=${lang}
        export YEAR=${yr} 
        ${EXPT}/kbp-nfold-trainer.sh \
            |& tee ${EXPT}/kbp-result/${lang}-${yr}-v${VERSION}.log
        sleep 30
    done
}


function nfold2single {
    base=${1}
    if [ -z ${base} ]
    then
        CRITICAL "no basename is provided"
    else
        for i in `seq 0 4`
        do
            ln -s \
                "${base}-case-sensitive.wordlist" \
                "${base}-${i}-case-sensitive.wordlist"
            ln -s \
                "${base}-case-insensitive.wordlist" \
                "${base}-${i}-case-insensitive.wordlist"
            ln -s "${base}.pkl" "${base}-${i}.pkl"
        done
    fi
}
