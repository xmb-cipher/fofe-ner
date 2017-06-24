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


MODEL_BASE=${KBP_MODEL_BASE:-"kbp-split"}
INFO "model-base     : ${MODEL_BASE}"

if [ ! -z ${PASS2ND} ]
then
    INFO "2nd-pass training"
    MODEL_BASE="${MODEL_BASE}-2nd"
fi

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



for f in `find -L ${train_path} -type f`
do
    x=$((RANDOM % 5))
    for i in `seq 0 4`
    do
        if [ ${x} -eq ${i} ]
        then
            ln -s ${f} ${dir}/split-${i}/${language}-eval-parsed/`basename ${f}`
        else
            # the loop is to control the ratio of kbp to iflytek
            for c in `seq 1 ${N_COPY:-1}`
            do
                ln -s ${f} ${dir}/split-${i}/${language}-train-parsed/copy-${c}-`basename ${f}`
            done
        fi
    done
done
INFO "kpb-data is processed"


if [ ! -z ${IFLYTEK+X} ]
then
    INFO "KBP_IFLYTEK == ${KBP_IFLYTEK}"
    iflytek_path=${dir}/iflytek
    cp -f -R -L ${KBP_IFLYTEK} ${iflytek_path}
    idx=0

    for f in `find -L ${iflytek_path} -type f`
    do
        fid=`printf %06d ${idx}`
        idx=$((idx + 1))
        next=$((RANDOM % 5))
        for i in `seq 0 4`
        do
            if [ ${next} -eq ${i} ]
            then
                next=$((RANDOM % 2))
                if [ ${next} -eq 0 ]
                then
                    ln -s ${f} ${dir}/split-${i}/${language}-eval-parsed/${fid}
                else
                    ln -s ${f} ${dir}/split-${i}/${language}-train-parsed/${fid}
                fi
            else
                ln -s ${f} ${dir}/split-${i}/${language}-train-parsed/${fid}
            fi
        done
    done
    INFO "iflytek-data is processed"
fi

# DEBUG
# while true; do sleep 128; done

INFO "training ... "

PROCESSED_DATA=$(for i in $(seq 0 4); do printf "${dir}/split-${i} "; done)
MODEL=$(for j in $(seq 0 4); do printf "${this_dir}/kbp-result/${MODEL_BASE}-${j} "; done)
LOG_FILE=$(for j in $(seq 0 4); do printf "${this_dir}/kbp-result/${MODEL_BASE}-${j}.log "; done)



# --cleanup option fails to remove folders
# parallel -env --link -S "image,music,audio,voice,language" \

CMD="parallel -env --link -j1"

if [ -z ${CUDA_VISIBLE_DEVICES} ]
then
    if [ -z ${SERVER_LIST} ] 
    then
        SERVER_LIST=`ServerList 5 | tr '\n' ',' | sed s'/,$//'`
        # SERVER_LIST="ea31,ea32,ea33,ea34,ea35"
    fi
    INFO "5 trainers are running on ${SERVER_LIST}"
    CMD="parallel -env --link -j5 -S ${SERVER_LIST} --basefile ${dir}"
fi

${CMD} \
    ${this_dir}/scripts/kbp-ed-trainer.sh \
    ::: ${embedding_path} \
    ::: $PROCESSED_DATA \
    ::: "--layer_size" ::: "512,512,512" \
    ::: "--n_batch_size" ::: "512" \
    ::: "--learning_rate" ::: "0.128" \
    ::: "--momentum" ::: "0.9" \
    ::: "--max_iter" ::: "${N_EPOCH:-256}" \
    ::: "--feature_choice" ::: "1023" \
    ::: "--dropout" \
    ::: "--char_alpha" ::: "0.8" \
    ::: "--word_alpha" ::: "0.5" \
    ::: "--language" ::: "${language}" \
    ::: "--model" :::+ $MODEL \
    ::: "--buffer_dir" ::: "${dir}" \
    ::: "--logfile" :::+ $LOG_FILE \
    ::: "--skip_test" \
    ::: "--version" ::: ${VERSION} \
    ::: "--average" \
    ${OPTION_2ND} 
 

# export CUDA_VISIBLE_DEVICES=0
INFO "evaluating ... "

${this_dir}/scripts/kbp-nfold-eval.py \
    ${eval_path} \
    ${this_dir}/kbp-result/${MODEL_BASE} \
    ${dir}/kbp-gaz.pkl \
    ${embedding_path} \
    ${dir}/combined |& tee ${dir}/report

tail -29 ${dir}/report | \
    mail -s "kbp-nfold-eval-${KBP_NFOLD_LANG}-${YEAR}-v${VERSION}" `whoami`@eecs.yorku.ca
    
