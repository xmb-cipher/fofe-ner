#!/bin/bash

# Instead of taking command line arguments, I put predefined hyper parameters here
# The hyper parameters should be fine-tuned in a non-cross-validation setting.

export CUDA_VISIBLE_DEVICES=0=

set -e
this_script=`which $0`
this_dir=`dirname ${this_script}`
. ${this_dir}/less-important/util.sh

if [ $# -lt 4 ] || [ $# -gt 5 ]
then
	printf ${KRED}
	printf "usage: %s <embedding-path> <kbp-path> <language> [iflytek-path]\n" $0 1>&2
	printf "    <embedding-path> : basename of word embedding, e.g. word2vec/gw256 \n" 1>&2
	printf "    <kbp-train-path> : directory containing parsed labeled data, e.g. processed-data/eng-train-parsed\n" 1>&2
	printf "    <kbp-eval-path>  : directory containing parsed labeled data, e.g. processed-data/eng-eval-parsed\n" 1>&2
	printf "    <languge>        : either eng, cmn or spa\n" 1>&2
	printf "    [iflytek-path]   : directory containing parsed iflytek data\n" 1>&2
	printf ${KNRM}
	exit 1
fi

INFO "$@"

embedding_path=$1 ; shift
train_path=$1 ; shift
eval_path=$1 ; shift
language=$1 ; shift
[ $# -eq 4 ] && iflytek_path=$1 &&
	INFO "iflytek data set is used in training"


dir=`mktemp -d`
trap "rm -rf ${dir}" EXIT
INFO "intermediate files are put in ${dir}"


cp -f -R ${train_path} ${dir}/kbp
cp -f ${train_path}/../kbp-gazetteer ${dir}/kbp-gazetteer
# head -1024 ${train_path}/../kbp-gazetteer > ${dir}/kbp-gazetteer
train_path=${dir}/kbp

cp -f -R ${eval_path} ${dir}/eval
eval_path=${dir}/eval

[ $# -eq 4 ] && cp -f -R ${iflytek_path} ${dir}/iflytek && iflytek_path=${dir}/iflytek


for i in `seq 0 4`
do
	dst="${dir}/split-${i}"
	mkdir -p ${dst}
	mkdir -p ${dst}/${language}-train-parsed
	mkdir -p ${dst}/${language}-eval-parsed
	ln -s ${dir}/kbp-gazetteer ${dst}/kbp-gazetteer
done
INFO "folders are created"



for f in `find ${train_path} -type f`
do
	# this line gives non-zero return, set -e complains
	# x=`expr ${RANDOM} % 5`
	x=`python -c "import random; print random.choice([0, 1, 2, 3, 4])"`

	for i in `seq 0 4`
	do
		if [ ${x} -eq ${i} ]
		then
			ln -s ${f} ${dir}/split-${i}/${language}-eval-parsed/`basename ${f}`
		else
			ln -s ${f} ${dir}/split-${i}/${language}-train-parsed/`basename ${f}`
		fi
	done
done
INFO "kpb-data is processed"


if [ $# -eq 4 ]
then
	for f in `find ${iflytek_path} -type f`
	do
		next=`expr ${RANDOM} % 5`
		for i in `seq 0 4`
		do
			if [ ${next} -eq ${i} ]
			then
				ln -s ${f} ${dir}/split-${i}/${language}-eval-parsed/`basename ${f}`
			else
				ln -s ${f} ${dir}/split-${i}/${language}-train-parsed/`basename ${f}`
			fi
		done
	done
	INFO "iflytek-data is processed"
fi


INFO "training ... "
for i in `seq 0 4`
do
	INFO ""
	INFO "training split-${i}"
	INFO ""

	${this_dir}/kbp-ed-trainer.py \
		${embedding_path} \
		${dir}/split-${i} \
		--layer_size 512,512,512 \
		--n_batch_size 512 \
		--learning_rate 0.128 \
		--momentum 0.9 \
		--max_iter 256 \
		--feature_choice 639 \
		--overlap_rate 0.36 \
		--disjoint_rate 0.09 \
		--dropout \
		--char_alpha 0.8 \
		--word_alpha 0.5 \
		--language ${language} \
		--model "kbp-result/kbp-split-${i}"
done


INFO "evaluating ... "

${this_dir}/kbp-nfold-eval.py \
	${eval_path} \
	kbp-result \
	${dir}/kbp-gazetteer \
	${embedding_path} \
	${dir}/combined
	
