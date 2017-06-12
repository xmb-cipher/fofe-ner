#!/bin/bash

# Instead of taking command line arguments, I put predefined hyper parameters here
# The hyper parameters should be fine-tuned in a non-cross-validation setting.

# export CUDA_VISIBLE_DEVICES=1

set -e
this_dir=$(cd $(dirname $0); pwd)
source ${this_dir}/path.sh
source ${this_dir}/config.sh
source ${this_dir}/scripts/utils.sh
export PYTHONPATH="${this_dir}"

# if [ $# -ne 2 ]
# then
# 	printf ${KRED}
# 	printf "usage: %s <embedding-path> <data-path>\n" $0 1>&2
# 	printf "    <embedding-path> : basename of word embedding, e.g. word2vec/reuters256 \n" 1>&2
# 	printf "    <data-path>      : directory containing eng.{train,testa,testb}\n" 1>&2
# 	printf ${KNRM}
# 	exit 1
# fi
# INFO "command executed: $@"

# embedding_path=$1 ; shift
# data_path=$1 ; shift

export embedding_path=${embedding_path:-${this_dir}/word2vec/reuters256}
export data_path=${data_path:-${this_dir}/processed-data/CoNLL2003}


dir=`mktemp -d`
trap "rm -rf ${dir}" EXIT
INFO "intermediate files are put in ${dir}"


cp -f -L ${data_path}/eng.testb ${dir}/eng.testb
cp -f -L ${data_path}/ner-lst ${dir}/ner-lst
for i in `seq 0 4`
do
	dst="${dir}/split-${i}"
	mkdir -p ${dst}
	ln -s ${dir}/eng.testb ${dst}/eng.testb
	ln -s ${dir}/ner-lst ${dst}/ner-lst
done

${this_dir}/scripts/conll2003-nfold-split.py ${data_path} ${dir}
INFO "Here's the file hierarchy"
tree ${dir} -L 2

INFO "training ..."

PROCESSED_DATA=$(for i in $(seq 0 4); do printf "${dir}/split-${i} "; done)
MODEL=$(for j in $(seq 0 4); do printf "${this_dir}/conll2003-model/${model:-split}-${j} "; done)
LOG_FILE=$(for j in $(seq 0 4); do printf "${this_dir}/conll2003-model/${model:-split}-${j}.log "; done)

SERVER_LIST=`ServerList | tail -5 | tr '\n' ',' | sed s'/,$//'`

INFO "5 trainers are running on ${SERVER_LIST}"

parallel -env --link -j5 \
	-S "${SERVER_LIST}" \
	--basefile ${dir} \
	${this_dir}/scripts/conll2003-ner-trainer.sh \
	::: ${embedding_path} \
	::: ${PROCESSED_DATA} \
	::: "--layer_size" ::: "512,512,512" \
	::: "--learning_rate" ::: "0.1024" \
	::: "--momentum" ::: "0.9" \
	::: "--max_iter" ::: "128" \
	::: "--feature_choice" ::: "767" \
	::: "--overlap_rate" ::: "0.36" \
	::: "--disjoint_rate" ::: "0.09" \
	::: "--dropout" \
	::: "--char_alpha" ::: "0.8" \
	::: "--word_alpha" ::: "0.5" \
	::: "--model" ::: ${MODEL} \
	::: "--buffer_dir" ::: "${dir}" \
	::: "--logfile" ::: ${LOG_FILE}



INFO "evaluating... "

${this_dir}/scripts/conll2003-nfold-eval.py \
	${extra_opt} \
	${this_dir}/conll2003-model/${model:-split} \
	${dir}/eng.testb \
	${dir}/predicted

INFO "final result"

${this_dir}/CoNLL2003eval.py \
	--threshold `head -1 ${dir}/predicted | cut -d' ' -f1` \
	--algorithm `head -1 ${dir}/predicted | cut -d' ' -f2` \
	--n_window `head -1 ${dir}/predicted | cut -d' ' -f3` \
	${dir}/eng.testb \
	<(awk "NR > 1" ${dir}/predicted) | \
${this_dir}/conlleval | tee ${dir}/result

awk 'NR == 2' ${dir}/result | rev | cut -d' ' -f1 | rev
