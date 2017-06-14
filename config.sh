#!/bin/bash

export EXPT=/eecs/research/asr/mingbin/ner-advance

export YEAR=${YEAR:-2016}
export N_COPY=1
export N_EPOCH=256
export VERSION=${VERSION:-1}

# KBP nfold config
export KBP_NFOLD_LANG=${KBP_NFOLD_LANG:-"eng"}
export KBP_MODEL_BASE=${KBP_MODEL_BASE:-${KBP_NFOLD_LANG}${YEAR}v${VERSION}}

if [ ${KBP_NFOLD_LANG} == "eng" ]
then
    export KBP_NFOLD_EMBED=${EXPT}/"word2vec/gigaword/gigaword128"
    export KBP_NFOLD_TRAIN=${EXPT}/"processed-data/KBP-EDL-${YEAR}/eng-train-parsed"
    export KBP_NFOLD_EVAL=${EXPT}/"processed-data/KBP-EDL-${YEAR}/eng-eval-parsed"
elif [ ${KBP_NFOLD_LANG} == 'cmn' ]
then
    export KBP_NFOLD_EMBED=${EXPT}/"word2vec/wiki-cmn"
    export KBP_NFOLD_TRAIN=${EXPT}/"processed-data/KBP-EDL-${YEAR}/cmn-train-parsed"
    export KBP_NFOLD_EVAL=${EXPT}/"processed-data/KBP-EDL-${YEAR}/cmn-eval-parsed"
elif [ ${KBP_NFOLD_LANG} == 'spa' ]
then
    export KBP_NFOLD_EMBED=${EXPT}/"word2vec/gigaword/spa-gw"
    export KBP_NFOLD_TRAIN=${EXPT}/"processed-data/KBP-EDL-${YEAR}/spa-train-parsed"
    export KBP_NFOLD_EVAL=${EXPT}/"processed-data/KBP-EDL-${YEAR}/spa-eval-parsed"
fi

if [ ! -z ${IFLYTEK+X} ]
then
    if [ ${KBP_NFOLD_LANG} == 'eng' ]
    then
        export KBP_IFLYTEK=${EXPT}/iflytek-clean-eng
        echo "KBP_IFLYTEK == ${KBP_IFLYTEK}"
        export N_COPY=4
        export N_EPOCH=64
    elif [ ${KBP_NFOLD_LANG} == 'cmn' ]
    then
        export KBP_IFLYTEK=${EXPT}/iflytek-clean-cmn
        export N_COPY=4
        export N_EPOCH=64
    else
        CRITICAL "IFLTTEK doesn't provide any annotation of the selected language"
        unset IFLYTEK
    fi
fi





