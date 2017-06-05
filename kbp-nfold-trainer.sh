#!/bin/bash

set -e
this_dir=$(cd $(dirname $0); pwd)
source ${this_dir}/path.sh
source ${this_dir}/config.sh


embedding_path=${KBP_NFOLD_EMBED}
train_path=${KBP_NFOLD_TRAIN}
eval_path=${KBP_NFOLD_EVAL}
language=${KBP_NFOLD_LANG}

INFO "embedding-path : ${embedding_path}"
INFO "train-path     : ${train_path}"
INFO "eval-path      : ${eval_path}"
INFO "language       : ${language}"


dir=`mktemp -d`
trap "rm -rf ${dir}" EXIT
INFO "intermediate files are put in ${dir}"


cp -f -R -L ${train_path} ${dir}/kbp
cp -f -L ${train_path}/../kbp-gaz.pkl ${dir}/kbp-gaz.pkl
train_path=${dir}/kbp

cp -f -R -L ${eval_path} ${dir}/eval
eval_path=${dir}/eval


for i in `seq 0 4`
do
    dst="${dir}/split-${i}"
    mkdir -p ${dst}
    mkdir -p ${dst}/${language}-train-parsed
    mkdir -p ${dst}/${language}-eval-parsed
    ln -s ${dir}/kbp-gaz.pkl ${dst}/kbp-gaz.pkl
done
INFO "folders are created"



for f in `find ${train_path} -type f`
do
    # this line gives non-zero return, set -e complains
    # x=`expr ${RANDOM} % 5`
    # x=`python -c "import random; print random.choice([0, 1, 2, 3, 4])"`
    x=$((RANDOM % 5))

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
        next=$((RANDOM % 5))
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

PROCESSED_DATA=$(for i in $(seq 0 4); do printf "${dir}/split-${i} "; done)
MODEL=$(for j in $(seq 0 4); do printf "${this_dir}/kbp-result/kbp-split-${j} "; done)
LOG_FILE=$(for j in $(seq 0 4); do printf "${this_dir}/kbp-result/kbp-split-${j}.log "; done)
SERVER_LIST=`ServerList | tail -5 | tr '\n' ',' | sed s'/,$//'`

INFO "5 trainers are running on ${SERVER_LIST}"

# --cleanup option fails to remove folders
# parallel -env --link -S "image,music,audio,voice,language" \
parallel -env --link -j5 \
    -S "${SERVER_LIST}" \
    --basefile ${dir} \
    ${this_dir}/scripts/kbp-ed-trainer.sh \
    ::: ${embedding_path} \
    ::: $PROCESSED_DATA \
    ::: "--layer_size" ::: "512,512,512" \
    ::: "--n_batch_size" ::: "512" \
    ::: "--learning_rate" ::: "0.128" \
    ::: "--momentum" ::: "0.9" \
    ::: "--max_iter" ::: "256" \
    ::: "--feature_choice" ::: "1023" \
    ::: "--dropout" \
    ::: "--char_alpha" ::: "0.8" \
    ::: "--word_alpha" ::: "0.5" \
    ::: "--language" ::: "${language}" \
    ::: "--model" :::+ $MODEL \
    ::: "--buffer_dir" ::: "${dir}" \
    ::: "--logfile" :::+ $LOG_FILE
 

INFO "evaluating ... "

${this_dir}/scripts/kbp-nfold-eval.py \
    ${eval_path} \
    ${this_dir}/kbp-result \
    ${dir}/kbp-gaz.pkl \
    ${embedding_path} \
    ${dir}/combined |& tee ${dir}/report

tail -32 ${dir}/report | \
    mail -s "kbp-nfold-eval" `whoami`@eecs.yorku.ca
    
