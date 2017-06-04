#!/bin/bash

export EXPT=/eecs/research/asr/mingbin/ner-advance

# KBP nfold config
export KBP_NFOLD_LANG=${KBP_NFOLD_LANG:-"eng"}
if [ ${KBP_NFOLD_LANG} == "eng" ]
then
    export KBP_NFOLD_EMBED=${EXPT}/"word2vec/gigaword/gigaword128"
    export KBP_NFOLD_TRAIN=${EXPT}/"processed-data/eng-train-parsed"
    export KBP_NFOLD_EVAL=${EXPT}/"processed-data/eng-eval-parsed"
fi
