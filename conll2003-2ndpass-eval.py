#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import numpy, argparse, logging, time, cPickle, codecs, os, getpass, sys

logger = logging.getLogger( __name__ )


if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level = logging.INFO )

    parser = argparse.ArgumentParser()

    parser.add_argument( 'model1st', type = str, 
                         help = 'basename of model trained for 1st pass' )
    parser.add_argument( 'model2nd', type = str,
                         help = 'basename of model trained for 2nd pass' )
    parser.add_argument( 'testb', type = str, 
                         help = 'CoNLL2003 evaluation set' )
    parser.add_argument( '--buf_dir', type = str, default = None,
                         help = 'writable directory buffering intermediate results' )
    parser.add_argument( '--nfold', action = 'store_true', default = False,
                         help = 'load 5 models if set' )

    args = parser.parse_args()
    logger.info( str(args) + '\n' )

    ################################################################################

    if args.buf_dir is None:
        buf_dir = os.path.join('/tmp', getpass.getuser() )
        if not os.path.exists( buf_dir ):
            os.makedirs( buf_dir )
    else:
        buf_dir = args.buf_dir

    ################################################################################

    from fofe_mention_net import *

    ################################################################################

    gazetteer_path = os.path.join( os.path.dirname( __file__ ),
                                   'conll2003-model', 'ner-list' )
    conll2003_gazetteer = gazetteer( gazetteer_path )

    ################################################################################
    ########## compute 1st-past result
    ################################################################################

    with open( '%s.config' % args.model1st, 'rb' ) as fp:
        config1 = cPickle.load( fp )
    logger.info( config1.__dict__ )
    logger.info( 'config1st loaded' )

    ################################################################################

    # TODO, integrate wordlist and model basename
    numericizer1 = vocabulary( os.path.join( os.path.dirname(__file__),
                                             'conll2003-model',
                                             'reuters256-case-insensitive.wordlist' ),
                               config1.char_alpha, False )
    numericizer2 = vocabulary( os.path.join( os.path.dirname(__file__),
                                             'conll2003-model',
                                             'reuters256-case-sensitive.wordlist' ),
                               config1.char_alpha, True )

    logger.info( 'vocabulary loaded' )

    ################################################################################

    test  = batch_constructor( CoNLL2003( args.testb ), 
                               numericizer1, numericizer2, 
                               gazetteer = conll2003_gazetteer, 
                               alpha = config1.word_alpha, 
                               window = config1.n_window )
    logger.info( 'test: ' + str(test) )
    logger.info( 'data set loaded' )

    ################################################################################

    mention_net = fofe_mention_net( config1 )
    mention_net.fromfile( args.model1st )
    logger.info( 'model loaded' )

    ################################################################################

    print1, predicted1st = [], os.path.join( buf_dir, 'predict1st' )
    for example in test.mini_batch_multi_thread( 
                        2560 if config1.feature_choice & (1 << 9) > 0 else 2560, 
                        False, 1, 1, config1.feature_choice ):

        _, pi, pv = mention_net.eval( example )
        print1.append( numpy.concatenate( 
                           ( example[-1].astype(numpy.float32).reshape(-1, 1),
                             pi.astype(numpy.float32).reshape(-1, 1),
                             pv ), axis = 1 ) )
        
    print1 = numpy.concatenate( print1, axis = 0 )
    numpy.savetxt( predicted1st, print1, 
                   fmt = '%d  %d' + '  %f' * (config1.n_label_type + 1) )

    logger.info( 'evaluation set passed first time' )

    pp = PredictionParser( SampleGenerator( args.testb ), predicted1st, config1.n_window )

    output1st = os.path.join(buf_dir, 'output1st')
    with open( output1st, 'wb' ) as out1st:
        _, _, _, info = evaluation( pp, config1.threshold, config1.algorithm,
                                    conll2003out = out1st,
                                    sentence_iterator = SentenceIterator( args.testb ) )
    logger.info( '\n' + info )
    logger.info( 'first-round output generated' )
    del mention_net

    ################################################################################
    ########## compute 2nd-pass result
    ################################################################################

    with open( '%s.config' % args.model2nd, 'rb' ) as fp:
        config2 = cPickle.load( fp )
    config2.is_2nd_pass = True
    assert config2.n_window == config1.n_window, 'inconsisitent window size'
    logger.info( config2.__dict__ )
    logger.info( 'config2nd loaded' )

    ################################################################################

    # TODO, integrate wordlist and model basename
    numericizer1 = vocabulary( os.path.join( os.path.dirname(__file__),
                                             'conll2003-model',
                                             'reuters256-case-insensitive.wordlist' ),
                               config2.char_alpha, False,
                               n_label_type = config2.n_label_type )
    numericizer2 = vocabulary( os.path.join( os.path.dirname(__file__),
                                             'conll2003-model',
                                             'reuters256-case-sensitive.wordlist' ),
                               config2.char_alpha, True,
                               n_label_type = config2.n_label_type )
    logger.info( 'vocabulary loaded' )

    ################################################################################

    test  = batch_constructor( CoNLL2003( output1st ), 
                               numericizer1, numericizer2, 
                               gazetteer = conll2003_gazetteer, 
                               alpha = config2.word_alpha, 
                               window = config2.n_window,
                               is2ndPass = True )
    logger.info( 'test: ' + str(test) )
    logger.info( 'data set loaded' )

    ################################################################################

    mention_net = fofe_mention_net( config2 )
    mention_net.fromfile( args.model2nd )
    logger.info( 'model loaded' )

    ################################################################################

    print2, predicted2nd = [], os.path.join( buf_dir, 'predict2nd' )
    for example in test.mini_batch_multi_thread( 
                        2560 if config1.feature_choice & (1 << 9) > 0 else 2560, 
                        False, 1, 1, config1.feature_choice ):

        _, pi, pv = mention_net.eval( example )
        print2.append( numpy.concatenate( 
                           ( example[-1].astype(numpy.float32).reshape(-1, 1),
                             pi.astype(numpy.float32).reshape(-1, 1),
                             pv ), axis = 1 ) )
        
    print2 = numpy.concatenate( print2, axis = 0 )
    numpy.savetxt( predicted2nd, print2, 
                   fmt = '%d  %d' + '  %f' * (config2.n_label_type + 1) )

    logger.info( 'evaluation set passed second time' )

    pp = PredictionParser( SampleGenerator( args.testb ), predicted2nd, config2.n_window )

    output2nd = os.path.join(buf_dir, 'output2nd')
    with open( output2nd, 'wb' ) as out2nd:
        _, _, _, info = evaluation( pp, config2.threshold, config2.algorithm,
                                    conll2003out = out2nd,
                                    sentence_iterator = SentenceIterator( args.testb ) )
    logger.info( '\n' + info )
    logger.info( 'second-round output generated' )
    del mention_net

    ################################################################################
    ########## combine 1st-pass and 2nd-pass result in terms of raw probability
    ################################################################################

    print3 = print2.copy()
    print3[:,2:] += print2[:,2:]
    print3[:,2:] /= 2
    threshold = (config1.threshold + config2.threshold) / 2

    predicted3rd = os.path.join( buf_dir, 'predict3rd' )
    numpy.savetxt( predicted3rd, print3, 
                   fmt = '%d  %d' + '  %f' * (config2.n_label_type + 1) )

    pp = PredictionParser( SampleGenerator( args.testb ), predicted3rd, config2.n_window )

    output3rd = os.path.join(buf_dir, 'output3rd')
    with open( output2nd, 'wb' ) as out3rd:
        _, _, _, info = evaluation( pp, threshold, config2.algorithm,
                                    conll2003out = out3rd,
                                    sentence_iterator = SentenceIterator( args.testb ) )
    logger.info( '\n' + info )
    logger.info( '1st-pass and 2nd-pass combined at probability level' )
