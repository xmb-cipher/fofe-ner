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
	result_file=${tmp_dir}/`printf result%04d ${x}`
	conll2003-nfold-trainer.sh \
			word2vec/reuters256 \
			processed-data |& tee ${result_file}
	info=`tail -7 ${result_file}`
	f1=`tail -1 ${result_file}`

	if [ `echo "${f1} > ${best_f1}" | bc` -eq 1 ]
	then
		best_f1=${f1}
		mv ${this_dir}/conll2003-model/${model:-split}-* ${best_dir}/
		echo "${info}"
		echo "f1: ${info}" | \
			mail -s "best-f1: ${best_f1} ${extra_opt}" `whoami`@eecs.yorku.ca
	fi
done

