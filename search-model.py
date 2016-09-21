#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

"""
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : kbp-ed-trainer.py
Last Update : Jul 26, 2016
Description : N/A
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
"""

import logging, argparse
from itertools import product, chain
from subprocess import Popen, PIPE, call

logger = logging.getLogger( __name__ )

logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                     level = logging.INFO )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--iflytek', action = 'store_true', default = False )
    args = parser.parse_args()

    if not args.iflytek:
        search_space = chain( product( [ 0.032, 0.064, 0.128 ],
                                       [ 32, 64, 128, 256, 512 ],
                                       [ 128, 256 ] ),
                              [ (0.016, 1, 32 ) ] )
    else:
        search_space = chain( product( [ 0.064, 0.128 ],
                                       [ 128, 256, 512 ],
                                       [ 16, 32 ] ),
                              [ (0.016, 1, 8 ) ] )

    for learning_rate, n_batch_size, max_iter in search_space:
        logger.info( '=' * 72 )
        logger.info( '%.4f %3d %2d' % (learning_rate, n_batch_size, max_iter) )
        logger.info( '=' * 72 + '\n' )

        cmd = 'kbp-system.py word2vec/reuters128 processed-data ' + \
                    '--dropout ' + \
                    '--max_iter=%d ' % max_iter + \
                    '--learning_rate=%f ' % learning_rate + \
                    '--n_batch_size=%d ' % n_batch_size + \
                    '--feature_choice=511 '

        if args.iflytek:
            cmd += '--model=include-iflytek --iflytek'
        else:
            cmd += '--model=exclude-iflytek'

        process = Popen( cmd, shell = True , stdout = PIPE, stderr = PIPE)
        (out, err) = process.communicate()
        exit_code = process.wait()

        logger.info( err )

