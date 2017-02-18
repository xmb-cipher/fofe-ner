#!/bin/bash

# Instead of taking command line arguments, I put predefined hyper parameters here
# The hyper parameters should be fine-tuned in a non-cross-validation setting.

export CUDA_VISIBLE_DEVICES=1

set -e
this_script=`which $0`
this_dir=`dirname ${this_script}`
. ${this_dir}/less-important/util.sh

if [ $# -ne 2 ] 
then
	printf ${KRED}
	printf "usage: %s <embedding-path> <data-path>\n" $0 1>&2
	printf "    <embedding-path> : basename of word embedding, e.g. word2vec/reuters256 \n" 1>&2
	printf "    <data-path>      : directory containing eng.{train,testa,testb}\n" 1>&2
	printf ${KNRM}
	exit 1
fi
INFO "command executed: $@"

embedding_path=$1 ; shift
data_path=$1 ; shift


dir=`mktemp -d`
trap "rm -rf ${dir}" EXIT
INFO "intermediate files are put in ${dir}"

cp -f ${data_path}/eng.testb ${dir}/eng.testb
cp -f ${data_path}/ner-lst ${dir}/ner-lst
for i in `seq 0 4`
do
	dst="${dir}/split-${i}"
	mkdir -p ${dst}
	ln -s ${dir}/eng.testb ${dst}/eng.testb
	ln -s ${dir}/ner-lst ${dst}/ner-lst
done

${this_dir}/conll2003-nfold-split.py ${data_path} ${dir}
INFO "Here's the file hierarchy"
tree ${dir} -L 2


for i in `seq 0 4`
do
	INFO ""
	INFO "training split-${i}"
	INFO ""

	${this_dir}/conll2003-ner-trainer.py \
		${embedding_path} \
		${dir}/split-${i} \
		--layer_size 512,512,512 \
		--n_batch_size 512 \
		--learning_rate 0.128 \
		--momentum 0.9 \
		--max_iter 64 \
		--feature_choice 639 \
		--overlap_rate 0.36 \
		--disjoint_rate 0.09 \
		--dropout \
		--char_alpha 0.8 \
		--word_alpha 0.5 \
		--model "split-${i}" \
		--buffer_dir ${dir}
done


INFO "evaluating... "

${this_dir}/conll2003-nfold-eval.py ${dir}/eng.testb ${dir}/predicted

INFO "final result"

${this_dir}/CoNLL2003eval.py \
	--threshold `head -1 ${dir}/predicted | cut -d' ' -f1` \
	--algorithm `head -1 ${dir}/predicted | cut -d' ' -f2` \
	--n_window `head -1 ${dir}/predicted | cut -d' ' -f3` \
	${dir}/eng.testb \
	<(awk "NR > 1" ${dir}/predicted) | \
${this_dir}/conlleval | tee ${dir}/result

awk 'NR == 2' ${dir}/result | rev | cut -d' ' -f1 | rev
