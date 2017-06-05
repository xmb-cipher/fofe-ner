#!/bin/bash

export EXPT=/eecs/research/asr/mingbin/ner-advance
export YEAR=${YEAR:-2016}

# KBP nfold config
export KBP_NFOLD_LANG=${KBP_NFOLD_LANG:-"eng"}
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

