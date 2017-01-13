#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

"""
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : numericize-reuters-ner.py
Last Update : Apr 1, 2016
Description : Written to evaluate NER
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
"""

"""
In the labelled files, only the following labels are present

B-LOC
B-MISC
B-ORG
I-LOC
I-MISC
I-ORG
I-PER
O

In this code, we will narrow it down into 5 classes, 
PER     persons, 
LOC     locations, 
ORG     organizations
MISC    names of miscellaneous entities not belonging to the previous 3 groups
O       unknown
"""


import numpy, argparse, logging, cPickle #, pandas
from gigaword2feature import *

logger = logging.getLogger()
#pandas.set_option('display.max_columns', 1024)


if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level= logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument( 'ner_file', type = str, help = 'labeld data set' )
    parser.add_argument( 'trainer_output', type = str, help = 'output from ner-trainer-conll2003.py' )
    parser.add_argument( '--n_window', type = int, default = 7, help = 'maximum length of ner candidate' )
    parser.add_argument( '--threshold', type = float, default = 0.5, 
                          help = 'probability under cut-off is still not considered as an ner' )
    parser.add_argument( '--algorithm', type = int, default = 1, 
                          help = '{1: highest score first, 2: longest coverage first, 3: highest score first after subsumssion}' )
    parser.add_argument( '--reinterpret_threshold', type = float, default = 0,
                          help = 'if argmax == O and P(O) < reinterpret_threshold, pick the 2nd highest' )
    parser.add_argument( '--config', type = str, default = None, 
                          help = "fofe-mention-net's config" )

    args = parser.parse_args() 
    ner_file = args.ner_file
    trainer_output = args.trainer_output
    n_window = args.n_window
    threshold = args.threshold
    algorithm = args.algorithm
    reinterpret = args.reinterpret_threshold

    customized_threshold = None
    if args.config:
        with open( args.config, 'rb' ) as fp:
            config = cPickle.load( fp )
        customized_threshold = config.customized_threshold

    pp = PredictionParser( SampleGenerator( ner_file ), 
                           trainer_output, n_window, reinterpret )

    evaluation( pp, threshold, algorithm, 
                surpress_output = False, 
                sentence_iterator = SentenceIterator( ner_file ),
                decoder_callback = customized_threshold )

