#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

"""
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : conll2003-ner-evaluator.py
Last Update : Jul 25, 2016
Description : N/A
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ./LICENSE)
"""

import numpy, argparse, logging, time, cPickle, codecs, getpass

logger = logging.getLogger( __name__ )

def ReadSentences( filename ):
    with codecs.open( filename, 'rb', 'utf8' ) as in_file:
        for line in in_file:
            line = line.strip()
            if len(line) > 0:
                yield line


def WrapSentences( filename ):
    for line in ReadSentences( filename ):
        sent = line.split()
        for i,w in enumerate( sent ):
            sent[i] = u''.join( c if 0 <= ord(c) < 128 else chr(0) for c in list(w) )
        yield sent, [], [], []



if __name__ == '__main__':

    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level = logging.INFO )

    parser = argparse.ArgumentParser()
    parser.add_argument( 'model', type = str,
                         help = 'basename of the model, including model & model.{config,meta}' )
    parser.add_argument( 'input', type = str, 
                         help = 'input file, one tokenized sentence per line' )
    parser.add_argument( 'output', type = str,
                         help = 'output file, original sentences followed by offsets and mention types' )

    args = parser.parse_args()
    logger.info( str(args) + '\n' ) 

    from fofe_mention_net import *
    idx2ner = [ 'PER', 'LOC', 'ORG', 'MISC', 'O' ]

    ########## load model configuration ##########

    # config_path = os.path.join( os.path.dirname(__file__), 
    #                             'conll2003-model', 
    #                             'hopeless.config' )
    config_path = '%s.config' % args.model
    with open( config_path, 'rb' ) as fp:
        config = cPickle.load( fp )
    logger.info( config.__dict__ )
    logger.info( 'config loaded' )

    ########## load model parameters ##########

    # param_path = os.path.join( os.path.dirname(__file__), 
    #                            'conll2003-model',
    #                            'hopeless' )
    param_path = args.model
    mention_net = fofe_mention_net( config )
    mention_net.fromfile( param_path )
    logger.info( 'model loaded' )

    ########## load vocabulary ##########

    numericizer1 = vocabulary( os.path.join( os.path.dirname(__file__),
                                             'conll2003-model',
                                             'reuters256-case-insensitive.wordlist' ),
                               config.char_alpha, False )
    numericizer2 = vocabulary( os.path.join( os.path.dirname(__file__),
                                             'conll2003-model',
                                             'reuters256-case-sensitive.wordlist' ),
                               config.char_alpha, True )
    logger.info( 'vocabulary loaded' )

    ########## load gazetteer ##########

    gazetteer_path = os.path.join( os.path.dirname( __file__ ),
                                   'conll2003-model', 'ner-list' )
    conll2003_gazetteer = gazetteer( gazetteer_path )

    ########## translate user input into features ##########

    data = batch_constructor( WrapSentences( args.input ),
                              numericizer1, numericizer2, 
                              gazetteer = conll2003_gazetteer, 
                              alpha = config.word_alpha, 
                              window = config.n_window, 
                              n_label_type = config.n_label_type )
    logger.info( 'data: ' + str(data) + '\n' )

    ########## exhausively try all combination ##########

    buf_dir = os.path.join('/tmp', getpass.getuser() )
    if not os.path.exists( buf_dir ):
        os.makedirs( buf_dir )
    buf_path = os.path.join( buf_dir, 'eval-buffer' )

    with open( buf_path, 'wb' ) as buff_file:
        for example in data.mini_batch_multi_thread( 
                        256 if config.feature_choice & (1 << 9) > 0 else 1024,  
                        False, 1, 1, config.feature_choice ):
            _, pi, pv = mention_net.eval( example )

            # expcted has gargadge values
            for estimate, probability in zip( pi, pv ):
                print >> buff_file, '%d  %d  %s' % \
                        (-1, estimate, '  '.join( [('%f' % x) for x in probability.tolist()] ))

    ########## decode fofe-net's output ##########

    with codecs.open( args.output, 'wb', 'utf8' ) as out_file:
        for sent, (s, table, estimate, actual) in zip( ReadSentences( args.input ),
                PredictionParser( WrapSentences( args.input ),
                                  buf_path, config.n_window,
                                  n_label_type = config.n_label_type ) ):

            estimate = decode( s, estimate, table, config.threshold, config.algorithm ) 
            estimate = [ (b,e,idx2ner[c]) for b,e,c in sorted(estimate) ]

            to_write = sent + u'\n'
            if len(estimate) > 0:
                to_write += u'  '.join(str(x) for x in estimate) + u'\n'
            to_write += u'\n'
            out_file.write( to_write )

            logger.info( to_write.replace( u'\n', u'   ' ) )
