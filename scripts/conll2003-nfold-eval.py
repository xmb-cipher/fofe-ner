#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import numpy, argparse, logging, time, cPickle, codecs

logger = logging.getLogger( __name__ )

if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level = logging.INFO )

    parser = argparse.ArgumentParser()
    parser.add_argument( 'basename', type = str, help = 'basename of a 5fold-cross-validation model' )
    parser.add_argument( 'testb', type = str,
                          help = 'path to eng.testb of conll2003' )
    parser.add_argument( 'combined_out', type = str,
                         help = 'average probability' )
    parser.add_argument( '--is_2nd_pass', action = 'store_true', default = False )

    args = parser.parse_args()
    logger.info( str(args) + '\n' ) 

    from fofe_mention_net import *

    ########## load gazetteer ##########

    gazetteer_path = os.path.join( os.path.dirname( __file__ ),
                                   '../conll2003-model', 'ner-lst' )
    conll2003_gazetteer = gazetteer( gazetteer_path )
    # conll2003_gazetteer = [ set() for _ in xrange(10) ]

    ########## compute probability ##########

    algorithm = numpy.zeros( (4,), dtype = numpy.int32 )

    for i in xrange(5):
        ########## load config ##########
        # basename = os.path.join( os.path.dirname(__file__),
        #                          'conll2003-model', 'split-%d' % i )
        basename = '%s-%d' % (args.basename, i)
        config = mention_config()
        with open( '%s.config' % basename, 'rb' ) as fp:
             config.__dict__.update( cPickle.load( fp ).__dict__ )
        logger.info( config.__dict__ )
        logger.info( 'config of split-%d loaded' % i )


        ########## load vocabulary ##########

        nt = config.n_label_type if config.is_2nd_pass else 0
        numericizer1 = vocabulary( os.path.join( os.path.dirname(__file__),
                                                 '../conll2003-model',
                                                 'reuters256-case-insensitive.wordlist' ),
                                   config.char_alpha, False,
                                   n_label_type = nt )
        numericizer2 = vocabulary( os.path.join( os.path.dirname(__file__),
                                                 '../conll2003-model',
                                                 'reuters256-case-sensitive.wordlist' ),
                                   config.char_alpha, True,
                                   n_label_type = nt )
        logger.info( 'vocabulary loaded' )

        if i == 0:
            window = config.n_window
            threshold = config.threshold
            algorithm[config.algorithm] += 1
        else:
            assert window == config.n_window, 'inconsistent window'
            threshold += config.threshold

        ########## load network ##########

        mention_net = fofe_mention_net( config )
        mention_net.fromfile( basename )
        logger.info( 'model of split-%d loaded' % i )

        ########## load testb ##########

        test = batch_constructor( CoNLL2003( args.testb ), 
                                  numericizer1, numericizer2, 
                                  gazetteer = conll2003_gazetteer, 
                                  alpha = config.word_alpha, 
                                  window = config.n_window,
                                  is2ndPass = args.is_2nd_pass )
        logger.info( 'testb loaded' )

        ########## compute probability ##########

        target_i, probability_i = [], []
        for example in test.mini_batch_multi_thread( 
                            256 if config.feature_choice & (1 << 9) > 0 else 1024, 
                            False, 1, 1, config.feature_choice ):
            _, _, pv = mention_net.eval( example )
            for e, p in zip( example[-1], pv ):
                target_i.append( e )
                probability_i.append( p )

        # the destructor of fofe_mention_net closes tf session
        # there is no guarantee GC does that immediately
        del mention_net

        target_i = numpy.asarray( target_i, dtype = numpy.int32 )
        probability_i = numpy.asarray( probability_i, dtype = numpy.float32 )
        if i == 0:
            target = target_i
            probability = probability_i
        else:
            assert( numpy.array_equal( target, target_i ) )
            probability += probability_i

    probability /= 5
    estimate = probability.argmax( axis = 1 )
    threshold /= 5
    algorithm = algorithm.argmax()

    ########## write probability ##########

    with open( args.combined_out, 'wb' ) as fp:
        print >> fp, '%f %d %d' % (threshold, algorithm, window)
        for t, e, p in zip( target, estimate, probability ):
            print >> fp, '%d  %d  %s' % \
                    (t, e, '  '.join( [('%f' % x) for x in p.tolist()] ))
