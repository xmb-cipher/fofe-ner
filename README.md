A FOFE-based Local Detection Approach for Named Entity Recognition and Mention Detection
==========

This repository implements [A FOFE-based Local Detection Approach for Named Entity Reconigtion and Mention Detection](https://arxiv.org/abs/1611.00801) with [TensorFlow](https://www.tensorflow.org/). It ranks 2nd in the [Entity Discovery and Linking (EDL)](http://nlp.cs.rpi.edu/kbp/2016/) in [TAC Knowledge Base Population (KBP) 2016](https://tac.nist.gov//2016/KBP/). 


## install.sh
Most scripts hard codes the paths to pre-trained word embeddings as well as training, development and test sets. ```install.sh``` organizes directories by establishing symbolic links. 


## skipgram-trainer.py & cmn-word-avg.py
A wrapper of [gensim](https://radimrehurek.com/gensim/) skip-gram. 

* When run on a Chinese dataset, the user needs to indicate whether word embedding or character embedding is desired. 
* When run on an English/Spanish dataset, the user needs to indicate whether the desired word embedding is case-sensitive or case-insensitive. 
* Words containing number are mapped to some predified labels, e.g. ```<date-value>```, ```<time-value>```, ```<phone-number>```, etc.

Because the detector is character-based in Chinese, each character is represented by the concatenation of character embedding and the word embedding where the character is from. Such word embedding is divided by the length of the number of characters. 


## setup.py & source/gigaword2feature.pyx
```source/gigaword2feature.pyx``` is written with [Cython](http://cython.org/). It is compiled by
```bash
python setup.py build_ext -i
```
and ```gigaword2feature.so``` will be generated in the root directory. 

Here are the basic ideas:
