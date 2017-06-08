#!/cs/local/bin/bash

set -e

export THIS_DIR=$(cd $(dirname $0); pwd)
source ${THIS_DIR}/scripts/utils.sh


if [ $# -ne 3 ]
then
	CRITICAL "Incorrect command-line argument(s)!"
	printf "Usage: $0 <model> <source> <gold> \n" 1>&2
	printf "    <model>  : a trained model returned by kbp-system.py \n" 1>&2
	printf "    <source> : directory containing labelless files returned by kbp-xml-parser.py \n" 1>&2
	exit 1
fi

MODEL=$1
IN_DIR=$2
GOLD=$3
INFO "MODEL == ${MODEL}"
INFO "IN_DIR == ${IN_DIR}"

# a sub-shell switches folder; must be absolute path
GOLD_DIR=$(cd $(dirname $GOLD); pwd)
INFO "GOLD_DIR == ${GOLD_DIR}"

GOLD=${GOLD_DIR}/`basename $3`
INFO "GOLD == ${GOLD}"

export PYTHONPATH="${THIS_DIR}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-""}
INFO "CUDA_VISIBLE_DEVICES == '${CUDA_VISIBLE_DEVICES}'"

BUFFER_DIR=`mktemp -d`
INFO "BUFFER_DIR == ${BUFFER_DIR}"
# trap "rm -rf ${BUFFER_DIR} > /dev/null" EXIT
mkdir -p ${BUFFER_DIR}/labeled
mkdir -p ${BUFFER_DIR}/result
mkdir -p ${BUFFER_DIR}/report

${THIS_DIR}/scripts/kbp-ed-evaluator.py \
	${MODEL} \
	${IN_DIR} \
	${BUFFER_DIR}/labeled \
	--nfold
INFO "Labels generated."

${THIS_DIR}/scripts/reformat.py \
	${BUFFER_DIR}/labeled \
	${BUFFER_DIR}/result/`basename ${MODEL}`.tsv
INFO "Labels reformated."

(	cd /eecs/research/asr/mingbin/ner-data/neleval;
	scripts/run_tac16_evaluation.sh \
		${GOLD} \
		${BUFFER_DIR}/result \
		${BUFFER_DIR}/report \
		4	)
INFO "Reported generated.\n"

INFO "Report:"
head -1 ${BUFFER_DIR}/report/*.tab
tail -1 ${BUFFER_DIR}/report/*.tab | cut -f2-

INFO "Details:"
cat ${BUFFER_DIR}/report/*.evaluation

