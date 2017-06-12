#!/bin/bash

set -e
this_script=`which $0`
this_dir=`dirname ${this_script}`
. ${this_dir}/less-important/util.sh


best_dir=${this_dir}/conll2003-model/best-model
[ ! -d ${best_dir} ] && mkdir -p ${best_dir}

tmp_dir=`mktemp -d`
trap "rm -rf ${tmp_dir}" EXIT

best_f1=${best_f1:-0}

for (( x = 0; ; x++ ))
do
	INFO "BEST SOR FAR: ${best_f1}"

	result_file=${tmp_dir}/`printf result%04d ${x}`

	conll2003-ner-trainer.py \
		word2vec/reuters256 \
		${data:-processed-data} \
		--layer_size 512,512,512 \
		--n_batch_size 512 \
		--learning_rate 0.1024 \
		--momentum 0.9 \
		--max_iter 128 \
		--feature_choice 767 \
		--overlap_rate 0.36 \
		--disjoint_rate 0.09 \
		--dropout \
		--char_alpha 0.8 \
		--word_alpha 0.5 \
		--gpu_fraction 0.48 \
		--buffer_dir ${tmp_dir} \
		--model ${model:-1st-pass-attempt} \
		${extra_opt} |& tee ${result_file}

	info="`tail -7 ${result_file} | head -5`"
	f1=`echo "${info}" | head -1 | rev | cut -d' ' -f1 | rev`

	echo "${info}"
	echo "f1 in this attempt: ${f1}"

	if [ `echo "${f1} > ${best_f1}" | bc` -eq 1 ]
	then
		best_f1=${f1}
		mv ${this_dir}/conll2003-model/${model:-1st-pass-attempt}* ${best_dir}/
		INFO "model copied to ${best_dir}"
		echo "${info}"
		echo "f1: ${info}" | \
			mail -s "best-f1: ${best_f1} ${model} ${extra_opt}" `whoami`@eecs.yorku.ca
	fi
done

