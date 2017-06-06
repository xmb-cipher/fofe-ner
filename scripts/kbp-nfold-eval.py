#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import numpy, argparse, logging, time, cPickle, codecs
from itertools import product, chain

logger = logging.getLogger( __name__ )


if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level = logging.INFO )

    parser = argparse.ArgumentParser()
    parser.add_argument( 'eval_parsed', type = str, 
                         help = 'e.g. processed-data/eng-eval-parsed' )
    parser.add_argument( 'model_dir', type = str,
                         help = 'e.g. kbp-result' )
    parser.add_argument( 'gazetteer', type = str, 
                         help = 'e.g. processed-data/kbp-gaz.pkl' )
    parser.add_argument( 'embedding', type = str,
                         help = 'e.g. word2vec/gw256' )
    parser.add_argument( 'combined_out', type = str )

    args = parser.parse_args()
    logger.info( str(args) + '\n' ) 

    from fofe_mention_net import *

    ########## load gazetteer ##########

    try:
        with open( args.gazetteer, 'rb' ) as fp:
            kbp_gazetteer = cPickle.load( fp )
    except:
        kbp_gazetteer = [ set() for _ in xrange(16) ]

    ########## compute probability ##########

    threshold = numpy.zeros( (2, ), dtype = numpy.float32 )
    algorithm = {}

    for i in xrange(5):
        ########## load config ##########

        basename = os.path.join( os.path.dirname(__file__),
                                 '../kbp-result', 'kbp-split-%d' % i )
        with open( '%s.config' % basename, 'rb' ) as fp:
            config = cPickle.load( fp )
        logger.info( config.__dict__ )
        logger.info( 'config of split-%d loaded' % i )

        if i == 0:
            window = config.n_window
            label_type = config.n_label_type
        else:
            assert window == config.n_window, 'inconsistent window'
            assert label_type == config.n_label_type, 'inconsistent label types'
        threshold += numpy.asarray( config.threshold, dtype = numpy.float32 )
        config.algorithm = tuple(config.algorithm)
        if config.algorithm in algorithm:
            algorithm[config.algorithm] += 1
        else:
            algorithm[config.algorithm] = 1

        ########## load test set ##########

        source = imap( lambda x: x[:4], LoadED( args.eval_parsed ) )

        if config.language != 'cmn':
            numericizer1 = vocabulary( args.embedding + '-case-insensitive.wordlist', 
                                       config.char_alpha, False )
            numericizer2 = vocabulary( args.embedding + '-case-sensitive.wordlist', 
                                       config.char_alpha, True )
        else:
            numericizer1 = chinese_word_vocab( args.embedding + '-char.wordlist' )
            numericizer2 = chinese_word_vocab( args.embedding + \
                                ('-avg.wordlist' if config.average else '-word.wordlist') )
        logger.info( 'numericizer initiated' )

        ########## load test set ##########

        test = batch_constructor( source,
                                  numericizer1, numericizer2, 
                                  gazetteer = kbp_gazetteer, 
                                  alpha = config.word_alpha, 
                                  window = config.n_window, 
                                  n_label_type = config.n_label_type,
                                  language = config.language )
        logger.info( 'test: ' + str(test) )
        logger.info( 'data set loaded' )

        ########## load network ##########

        mention_net = fofe_mention_net( config )
        mention_net.fromfile( basename )
        logger.info( 'model of split-%d loaded' % i )

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

        logger.info( 'model of split-%d has gone through test set' % i )

    probability /= 5
    estimate = probability.argmax( axis = 1 )

    ########## decode ##########

    threshold /= 6.25
    threshold = threshold.tolist()
    logger.info( 'threshold: %s' % str(threshold) )

    algo = None
    for a in algorithm:
        if algo is None:
            algo = a
        elif algorithm[a] > algorithm[algo]:
            algo = a
    algorithm = list(algo)
    logger.info( 'algorithm: %s' % str(algorithm) )


    with open( args.combined_out, 'wb' ) as fp:
        for t, e, p in zip( target, estimate, probability ):
            print >> fp, '%d  %d  %s' % \
                    (t, e, '  '.join( [('%f' % x) for x in p.tolist()] ))

    pp = [ p for p in PredictionParser( imap( lambda x: x[:4], LoadED( args.eval_parsed ) ),
                                        args.combined_out, 
                                        window,
                                        n_label_type = label_type ) ]

    _, _, best_dev_fb1, info = evaluation( pp, threshold, algorithm, True, 
                                           n_label_type = label_type )
    logger.info( '%s\n%s' % ('validation', info) ) 

    ########## check the true power of the model ##########

    idx2algo = { 1: 'highest-first', 2: 'longest-first', 3:'subsumption-removal'  }
    algo2idx = { 'highest-first': 1, 'longest-first': 2, 'subsumption-removal': 3 }
    best_dev_fb1, best_threshold, best_algorithm = 0, [0.5, 0.5], [1, 1]

    for algorithm in product( [1, 2], repeat = 2 ):
        algorithm = list( algorithm )
        name = [ idx2algo[i] for i in algorithm  ]
        for threshold in product( [ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ], repeat = 2 ):
            threshold = list( threshold )

            precision, recall, f1, info = evaluation( pp, threshold, algorithm, True,
                                                   n_label_type = label_type )
            if f1 > best_dev_fb1:
                best_dev_fb1, best_threshold, best_algorithm = f1, threshold, algorithm
                best_precision, best_recall = precision, recall
                best_info = info

    logger.info( 'cut-off: %s, algorithm: %-20s' % \
                         (str(best_threshold), str([ idx2algo[i] for i in best_algorithm ])) )
    logger.info( '%s\n%s' % ('test', best_info) ) 
