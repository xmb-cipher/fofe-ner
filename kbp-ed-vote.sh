#!/cs/local/bin/bash

set -e

export THIS_DIR=$(cd $(dirname $0); pwd)
source ${THIS_DIR}/scripts/utils.sh


if [ $# -lt 3 ]
then
    CRITICAL "Incorrect command-line argument(s)!"
    printf "Usage: $0 <model> <source> <gold> \n" 1>&2
    printf "    <model>  : a trained model returned by kbp-system.py \n" 1>&2
    printf "    <source> : directory containing labelless files returned by kbp-xml-parser.py \n" 1>&2
    printf "    <gold>   : official solution \n" 1>&2
    exit 1
fi

MODEL=$1
IN_DIR=$2
GOLD=$3
NFOLD=$4
INFO "MODEL == ${MODEL}"
INFO "IN_DIR == ${IN_DIR}"

# a sub-shell switches folder; must be absolute path
GOLD_DIR=$(cd $(dirname $GOLD); pwd)
INFO "GOLD_DIR == ${GOLD_DIR}"

GOLD=${GOLD_DIR}/`basename $3`
INFO "GOLD == ${GOLD}"

export PYTHONPATH="${THIS_DIR}"
export CUDA_VISIBLE_DEVICES=''
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-""}
# INFO "CUDA_VISIBLE_DEVICES == '${CUDA_VISIBLE_DEVICES}'"

BUFFER_DIR=`mktemp -d`
INFO "BUFFER_DIR == ${BUFFER_DIR}"
trap "rm -rf ${BUFFER_DIR} > /dev/null" EXIT
mkdir -p ${BUFFER_DIR}/labeled
mkdir -p ${BUFFER_DIR}/result
mkdir -p ${BUFFER_DIR}/report

for i in $(seq 0 4)
do
    mkdir -p mkdir -p ${BUFFER_DIR}/labeled/${i}

    ${THIS_DIR}/scripts/kbp-ed-evaluator.py \
        ${MODEL}-${i} \
        ${IN_DIR} \
        ${BUFFER_DIR}/labeled/${i} \
        ${NFOLD}
    INFO "Labels of model-${i} generated."

    ${THIS_DIR}/scripts/reformat.py \
        ${BUFFER_DIR}/labeled/${i} \
        ${BUFFER_DIR}/result/${i}.tsv
    INFO "Labels for model-${i} reformated."
done


if [ ! -z ${RESULT_DIR} ]
then
    for i in $(seq 0 4)
    do
        cp -f -L \
            ${BUFFER_DIR}/result/${i}.tsv \
            ${RESULT_DIR}/${i}.tsv
    done
fi


${THIS_DIR}/scripts/vote.py ${BUFFER_DIR}/result


(   cd /eecs/research/asr/mingbin/ner-data/neleval;
    scripts/run_tac16_evaluation.sh \
        ${GOLD} \
        ${BUFFER_DIR}/result \
        ${BUFFER_DIR}/report \
        10   )
INFO "Reported generated.\n"

INFO "Report:"
cat ${BUFFER_DIR}/report/00report.tab
# head -n 1 ${BUFFER_DIR}/report/*.tab
# tail -n 1 ${BUFFER_DIR}/report/*.tab | cut -f2-

INFO "Details:"
for f in $(ls ${BUFFER_DIR}/report/*.evaluation)
do
    echo "::::::::::::::::"
    INFO "$(basename ${f})"
    echo "::::::::::::::::"
    cat ${f}
done

