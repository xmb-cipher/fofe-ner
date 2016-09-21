#!/cs/local/bin/bash

Color_Off="\033[0m"
Red="\033[0;31m"

function INFO() {
	printf ${Red}
	printf "[%s] %s\n" "`date`" "$@"
	printf ${Color_Off}
}


printf "Please enter the following paths. Default is used if it's skipped.\n\n"

read -p "word2vec directory, followed by [ENTER]: " word2vec
word2vec=${word2vec:-"/eecs/research/asr/mingbin/cleaner/word2vec"}
ln -s ${word2vec} "word2vec"
INFO "word2vec --> ${word2vec}"
echo

read -p "Directory containing parsed corpus and gazetteer, followed by [ENTER]: " parsed
parsed=${parsed:-"/eecs/research/asr/mingbin/ner-data/processed-data"}
ln -s ${parsed} "processed-data"
INFO "processed-data --> ${parsed}"
echo

read -p "Directory containing CoNLL2003's models, followed by [ENTER]: " conll_model
conll_model=${conll_model:-"/eecs/research/asr/mingbin/ner-data/conll2003-model"}
ln -s ${conll_model} "conll2003-model"
INFO "conll2003-model --> ${conll_model}"
echo 

read -p "Directory containing CoNLL2003's models, followed by [ENTER]: " conll_result
conll_result=${conll_model:-"/eecs/research/asr/mingbin/ner-data/conll2003-result"}
ln -s ${conll_result} "conll2003-result"
INFO "conll2003-result --> ${conll_result}"
echo 

read -p "Root directory of iFLYTEK's Chinese data, followed by [ENTER]: " iflytek_cmn
iflytek_cmn=${iflytek_cmn:-"/eecs/research/asr/mingbin/ner-data/iflytek-clean-cmn"}
ln -s ${iflytek_cmn} "iflytek-clean-cmn"
INFO "iflytek-clean-cmn --> ${iflytek_cmn}"
echo

read -p "Root directory of iFLYTEK's English data, followed by [ENTER]: " iflytek_eng
iflytek_eng=${iflytek_eng:-"/eecs/research/asr/mingbin/ner-data/iflytek-clean-eng"}
ln -s ${iflytek_eng} "iflytek-clean-eng"
INFO "iflytek-clean-eng --> ${ilfytek_eng}"
echo

read -p "Directory used to store logging, followed by [ENTER]: " ${log}
log=${log:-"/eecs/research/asr/mingbin/ner-data/log"}
ln -s ${log} "log"
INFO "log --> ${log}"
echo

read -p "Root directory of distant-supervision files, followd by [ENTER]: " ${dp}
dp=${dp:-"/local/scratch/mingbin/distant-supervision"}
ln -s ${dp} "distant-supervision"
INFO "distant-supervision --> ${dp}"
echo

printf "A buffer directory is created under tmp."
result_buff="/tmp/`whoami`/kbp-result"
mkdir ${result_buff}
ln -s ${result_buff} "kbp-result"
INFO "kbp-result --> ${result_buff}"
echo

python setup.py build_ext -i

