#!/bin/bash

export THIS_DIR=$(cd $(dirname $0); pwd)


export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export YEAR=2017


# export RAW_DIR=/local/scratch/mingbin/EDL-labelless
export RAW_DIR=/local/scratch/mingbin/kbp2017eval-selected

export MODEL_DIR=/local/scratch/mingbin/kbp2017models
mkdir -p ${MODEL_DIR}

# LABEL_DIR=/local/scratch/mingbin/kbp2017labels
export LABEL_DIR=/local/scratch/mingbin/kbp2017labels-selected
mkdir -p ${LABEL_DIR}

# TAB_DIR=/local/scratch/mingbin/kbp2017tabs
export TAB_DIR=/local/scratch/mingbin/kbp2017tabs-selected
mkdir -p ${TAB_DIR}


# 1. try to pick a GPU without user
# 2. if 1 fails, pick the GPU with most free memory
function pickGPU {
    SHOW_ID=$(
        nvidia-smi -q | \
        grep -i processes | \
        grep -in none | \
        cut -d':' -f1 | \
        tail -1
    )
    if [ -z ${SHOW_ID} ]
    then
        SHOW_ID=$(
            nvidia-smi -q | \
            grep Free | \
            awk 'NR % 2 == 1' | \
            tr -s ' ' | \
            cut -d' ' -f4 | \
            cat -n | \
            sort -k2 -nr | \
            head -n 1 | \
            cut -f1
        )
    fi
    DEVICE_ID=$((${SHOW_ID} - 1))
    [ $(hostname) == 'image' ] && DEVICE_ID=$((4 - ${SHOW_ID}))
    [ $(hostname) == 'text' ] && DEVICE_ID=$((4 - ${SHOW_ID}))
    [ $(hostname) == 'video' ] && DEVICE_ID=$(((${SHOW_ID} + 2) % 3))
    echo ${DEVICE_ID}
}
export -f pickGPU



########## TODO ############################################
# *.{wordlist,wubi} are copied and linked manually
# it should be automated
############################################################

###################
# MODEL PREPARATION
###################

export INCLUDE2015=true
for lang in spa cmn eng
do
    export KBP_NFOLD_LANG=${lang}
    export IFLYTEK=true

    export FOFE_EMBED=true
    ${THIS_DIR}/kbp-nfold-trainer.sh
    mkdir -p ${MODEL_DIR}/${lang}/fofe-embed-in2015
    mv ${THIS_DIR}/kbp-result/${lang}2017v1-* ${MODEL_DIR}/${lang}/fofe-embed-in2015

    unset FOFE_EMBED
    ${THIS_DIR}/kbp-nfold-trainer.sh
    mkdir -p ${MODEL_DIR}/${lang}/word2vec-in2015
    mv ${THIS_DIR}/kbp-result/${lang}2017v1-* ${MODEL_DIR}/${lang}/word2vec-in2015
done


unset INCLUDE2015
for lang in eng spa cmn
do
    export KBP_NFOLD_LANG=${lang}
    export IFLYTEK=true

    export FOFE_EMBED=true
    ${THIS_DIR}/kbp-nfold-trainer.sh
    mkdir -p ${MODEL_DIR}/${lang}/fofe-embed-ex2015
    mv -f ${THIS_DIR}/kbp-result/${lang}2017v1-* ${MODEL_DIR}/${lang}/fofe-embed-ex2015

    unset FOFE_EMBED
    ${THIS_DIR}/kbp-nfold-trainer.sh
    mkdir -p ${MODEL_DIR}/${lang}/word2vec-ex2015
    mv -f ${THIS_DIR}/kbp-result/${lang}2017v1-* ${MODEL_DIR}/${lang}/word2vec-ex2015
done


source ${THIS_DIR}/path.sh
for lang in spa cmn eng
do
    for embed in "fofe-embed" "word2vec"
    do
        for dataset in "in" "ex"
        do
            cd ${MODEL_DIR}/${lang}/${embed}-${dataset}2015
            nfold2single ${lang}2017v1 ${lang}
        done
    done
done
cd ${MODEL_DIR}

##################
# DATA PREPARATION
##################

source ${THIS_DIR}/path.sh
export CORPUS=/eecs/research/asr/Shared/KBP2017/Eval-2017/LDC2017E51_TAC_KBP_2017_Evaluation_Core_Source_Corpus
for lang in eng cmn spa
do
    mkdir ${RAW_DIR}/${lang}
    ${THIS_DIR}/kbp-xml-parser.py \
        ${CORPUS}/data/${lang} \
        ${RAW_DIR}/${lang} \
        --language ${lang} \
        --quote ${CORPUS}/docs/quote_regions.tsv
done


############
# ANNOTATION
############

function _annotate {
    export CUDA_VISIBLE_DEVICES=$(pickGPU)
    source ${THIS_DIR}/path.sh
    ${THIS_DIR}/scripts/kbp-ed-evaluator.py ${1} ${2} ${3}
}
export -f _annotate


function annotate {
    MODEL=${1}
    IN_DIR=${2}
    LABELED=${3}
    TAB=${4}

    MODELS=$(for i in $(seq 0 4); do printf "${MODEL}-%d " ${i}; done)
    LABELS=$(for i in $(seq 0 4); do printf "${LABELED}/%d " ${i}; done)

    for i in $(seq 0 4)
    do
        mkdir -p ${LABELED}/${i}
    done

    parallel --link -env -j3 -k -S : --sshdelay 10 \
        _annotate \
        ::: ${MODELS} \
        ::: ${IN_DIR} \
        ::: ${LABELS}

    for i in $(seq 0 4)
    do
        ${THIS_DIR}/scripts/reformat.py \
            ${LABELED}/${i} \
            ${TAB}/${i}.tsv
    done
}
export -f annotate


function annotate_by_lang {
    lang=${1}
    for embed in "fofe-embed" "word2vec"
    do
        for dataset in "in" "ex"
        do
            mkdir -p ${LABEL_DIR}/${lang}/${embed}-${dataset}2015
            mkdir -p ${TAB_DIR}/${lang}/${embed}-${dataset}2015

            annotate \
                ${MODEL_DIR}/${lang}/${embed}-${dataset}2015/${lang}2017v1 \
                ${RAW_DIR}/${lang} \
                ${LABEL_DIR}/${lang}/${embed}-${dataset}2015 \
                ${TAB_DIR}/${lang}/${embed}-${dataset}2015
        done
    done
}
export -f annotate_by_lang

parallel --link -env -j3 -k -S : --sshdelay 30 annotate_by_lang ::: eng cmn spa


##########
# EMSEMBLE
##########

for lang in eng cmn spa
do
    cnt=0
    rm -rf ${TAB_DIR}/${lang}/all
    mkdir -p ${TAB_DIR}/${lang}/all
    for f in $(find ${TAB_DIR}/${lang} -name *.tsv)
    do
        cp -f ${f} ${TAB_DIR}/${lang}/all/${cnt}.tsv
        cnt=$((cnt + 1))
    done

    model_cnt=$(ls ${TAB_DIR}/${lang}/all/*.tsv | wc -l)
    ${THIS_DIR}/scripts/vote.py ${TAB_DIR}/${lang}/all --model_cnt ${model_cnt}
done


##################
# SUMMISSION MERGE
##################

