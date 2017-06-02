#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import numpy, argparse, logging, time, cPickle, codecs, copy
from itertools import product, chain


# ================================================================================


if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                     level = logging.INFO )

    parser = argparse.ArgumentParser()
    parser.add_argument( 'basename', type = str )
    parser.add_argument( '--iflytek', action = 'store_true', default = False )

    args = parser.parse_args()
    logger.info( str(args) + '\n' ) 

    from fofe_mention_net import *

    # ================================================================================

    with open( args.basename + '.config', 'rb' ) as fp:
        config = cPickle.load( fp )
    logger.info( config.__dict__ )
    logger.info( 'configuration loaded' )

    valid_file = 'kbp-result/valid-tune-%s.predicted' % config.language
    test_file = 'kbp-result/test-tune-%s.predicted' % config.language

    assert config.language == 'spa' or args.iflytek

    if not os.path.exists( valid_file ) or \
            config.language == 'spa' or \
            not os.path.exists( test_file ):
        mention_net = fofe_mention_net( config )
        mention_net.fromfile( args.basename )
        logger.info( 'model loaded' )

        if config.language != 'cmn':
            numericizer1 = vocabulary( config.word_embedding + '-case-insensitive.wordlist', 
                                       config.char_alpha, False )
            numericizer2 = vocabulary( config.word_embedding + '-case-sensitive.wordlist', 
                                       config.char_alpha, True )
        else:
            numericizer1 = chinese_word_vocab( config.word_embedding + '-char.wordlist' )
            numericizer2 = chinese_word_vocab( config.word_embedding + \
                                ('-avg.wordlist' if config.average else '-word.wordlist') )
        logger.info( 'vocabulary loaded' )

        kbp_gazetteer = gazetteer( config.data_path + '/kbp-gazetteer' )

    # idx2ner = [ 'PER_NAM', 'PER_NOM', 'ORG_NAM', 'GPE_NAM', 'LOC_NAM', 'FAC_NAM', 'TTL_NAM', 'O'  ]
    idx2ner = [ 'PER_NAM', 'ORG_NAM', 'GPE_NAM', 'LOC_NAM', 'FAC_NAM',
                'PER_NOM', 'ORG_NOM', 'GPE_NOM', 'LOC_NOM', 'FAC_NOM',
                'O' ] 

    if not os.path.exists( valid_file ):
        # load 10% KBP test data
        source = imap( lambda x: x[1],
                       ifilter( lambda x : x[0] % 10 >= 9,
                       enumerate( imap( lambda x: x[:4], 
                                        LoadED( config.data_path + '/%s-eval-parsed' % config.language ) ) ) ) )

        # load 5% iflytek data
        if args.iflytek:
            source = chain( source, 
                            imap( lambda x: x[1],
                                  ifilter( lambda x : 90 <= x[0] % 100 < 95,
                                           enumerate( imap( lambda x: x[:4], 
                                                      LoadED( 'iflytek-clean-%s' % config.language ) ) ) ) ) )

        # istantiate a batch constructor
        valid = batch_constructor( source,
                                   numericizer1, numericizer2, gazetteer = kbp_gazetteer, 
                                   alpha = config.word_alpha, window = config.n_window, 
                                   n_label_type = config.n_label_type,
                                   language = config.language )
        logger.info( 'valid: ' + str(valid) )


    if config.language != 'spa' and not os.path.exists( test_file ):
        source = imap( lambda x: x[1],
                       ifilter( lambda x: x[0] % 100 >= 95,
                                enumerate( imap( lambda x: x[:4],
                                           LoadED( 'iflytek-clean-%s' % config.language ) ) ) ) )
        test = batch_constructor(  source,
                                   numericizer1, numericizer2, gazetteer = kbp_gazetteer, 
                                   alpha = config.word_alpha, window = config.n_window, 
                                   n_label_type = config.n_label_type,
                                   language = config.language )
        logger.info( 'test: ' + str(test) )


    # ================================================================================

    if not os.path.exists( 'kbp-result' ):
        os.makedirs( 'kbp-result' )


    ###############################################
    ########## go through validation set ##########
    ###############################################

    if not os.path.exists( valid_file ):
        with open( valid_file, 'wb' ) as valid_predicted:
            cost, cnt = 0, 0
            for example in valid.mini_batch_multi_thread( 
                            256 if config.feature_choice & (1 << 9 ) > 0 else 1024, 
                            False, 1, 1, config.feature_choice ):

                c, pi, pv = mention_net.eval( example )

                cost += c * example[-1].shape[0]
                cnt += example[-1].shape[0]
                for expected, estimate, probability in zip( example[-1], pi, pv ):
                    print >> valid_predicted, '%d  %d  %s' % \
                            (expected, estimate, '  '.join( [('%f' % x) for x in probability.tolist()] ))

            valid_cost = cost / cnt 
        logger.info( 'validation set iterated' )


    #########################################
    ########## go through test set ##########
    #########################################

    if config.language != 'spa' and not os.path.exists( test_file ):
        with open( test_file, 'wb' ) as test_predicted:
            cost, cnt = 0, 0
            for example in test.mini_batch_multi_thread( 
                            256 if config.feature_choice & (1 << 9 ) > 0 else 1024, 
                            False, 1, 1, config.feature_choice ):

                c, pi, pv = mention_net.eval( example )

                cost += c * example[-1].shape[0]
                cnt += example[-1].shape[0]
                for expected, estimate, probability in zip( example[-1], pi, pv ):
                    print >> test_predicted, '%d  %d  %s' % \
                            (expected, estimate, '  '.join( [('%f' % x) for x in probability.tolist()] ))

            test_cost = cost / cnt 
            logger.info( 'test set iterated' )


    ###################################################################################
    ########## exhaustively iterate 3 decodding algrithms with 0.x cut-off ############
    ###################################################################################


    # algo_list = ['highest-first', 'longest-first', 'subsumption-removal']
    idx2algo = { 1: 'highest-first', 2: 'longest-first', 3:'subsumption-removal'  }
    algo2idx = { 'highest-first': 1, 'longest-first': 2, 'subsumption-removal': 3 }

    source = imap( lambda x: x[1],
                   ifilter( lambda x : x[0] % 10 >= 9,
                   enumerate( imap( lambda x: x[:4], 
                                    LoadED( config.data_path + '/%s-eval-parsed' % config.language ) ) ) ) )
    if args.iflytek:
        source = chain( source, 
                        imap( lambda x: x[1],
                              ifilter( lambda x : 90 <= x[0] % 100 < 95,
                                       enumerate( imap( lambda x: x[:4], 
                                                  LoadED( 'iflytek-clean-%s' % config.language ) ) ) ) ) )

    # ================================================================================

    pp = [ p for p in PredictionParser( source,
                                        valid_file, 
                                        config.n_window,
                                        n_label_type = config.n_label_type ) ]

    algorithm = config.algorithm 
    threshold = config.threshold
    name = [ idx2algo[i] for i in algorithm  ]

    # ================================================================================

    _, _, best_dev_fb1, info = evaluation( pp, [0.5, 0.9], [2, 1], True, 
                                            n_label_type = config.n_label_type,
                                            decoder_callback = IndividualThreshold( [0.5] * 10, 0.5 ) )
    logger.info( '%s\n%s' % ('validation', info) ) 

    _, _, best_dev_fb1, info = evaluation( pp, [0.5, 0.5], [1, 1], True, 
                                            n_label_type = config.n_label_type )
    logger.info( '%s\n%s' % ('validation', info) ) 

    _, _, best_dev_fb1, info = evaluation( pp, threshold, algorithm, True, 
                                            n_label_type = config.n_label_type )
    logger.info( '%s\n%s' % ('validation', info) ) 

    # ================================================================================

    config.customized_threshold = IndividualThreshold( [threshold[0]] * 10, [threshold[1]] * 10 )

    # for algorithm in product( [1, 2], repeat = 2 ):
    #     algorithm = list( algorithm )

    for _ in xrange(3):
        for mt in xrange(10):
            for t in numpy.arange(0.1, 1, 0.1).tolist():
                it = copy.deepcopy( config.customized_threshold )
                it.outer[mt] = t

                precision, recall, f1, info = evaluation( pp, threshold, algorithm, True,
                                                          n_label_type = config.n_label_type,
                                                          decoder_callback = it )
                logger.info( 'validation -- precision: %f,  recall: %f,  fb1: %f' % (precision, recall, f1) )

                if f1 > best_dev_fb1:
                    best_dev_fb1, best_algorithm = f1, algorithm
                    best_precision, best_recall = precision, recall

                    # update threshold
                    config.customized_threshold = copy.deepcopy( it )

                    logger.info( 'algorithm: %-20s  outer: %s' % \
                                 (str([idx2algo[i] for i in algorithm]), str(it.outer)) )
                    logger.info( '%s\n%s' % ('validation', info) )

        for mt in xrange(10):
            for t in numpy.arange(0.1, 1, 0.1).tolist():
                it = copy.deepcopy( config.customized_threshold )
                it.inner[mt] = t

                precision, recall, f1, info = evaluation( pp, threshold, algorithm, True,
                                                          n_label_type = config.n_label_type,
                                                          decoder_callback = it )
                logger.info( 'validation -- precision: %f,  recall: %f,  fb1: %f' % (precision, recall, f1) )

                if f1 > best_dev_fb1:
                    best_dev_fb1, best_algorithm = f1, algorithm
                    best_precision, best_recall = precision, recall

                    # update threshold
                    config.customized_threshold = copy.deepcopy( it )

                    logger.info( 'algorithm: %-20s  inner: %s' % \
                                 (str([idx2algo[i] for i in algorithm]), str(it.inner)) )
                    logger.info( '%s\n%s' % ('validation', info) )

    logger.info( 'outer: ' + str(config.customized_threshold.outer) ) 
    logger.info( 'inner: ' + str(config.customized_threshold.inner) )

    # ================================================================================

    # for threshold in product( numpy.arange( max(threshold[0] - 0.1, 0.2), 
    #                                         min(threshold[0] + 0.2, 1.0),
    #                                         0.05 ).tolist(),
    #                           numpy.arange( max(threshold[1] - 0.1, 0.2), 
    #                                         min(threshold[1] + 0.2, 1.0),
    #                                         0.05 ).tolist() ):
    #     threshold = list( threshold )

    #     for customized_threshold in [ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]:
    #         customized = ORGcoverGPE( customized_threshold )

    #         precision, recall, f1, info = evaluation( pp, threshold, algorithm, True,
    #                                                   n_label_type = config.n_label_type,
    #                                                   decoder_callback = customized )
    #         logger.info( ('cut-off: %s, algorithm: %-20s' % (str(threshold), name)) + 
    #                       (', validation -- precision: %f,  recall: %f,  fb1: %f' % (precision, recall, f1)) )

    #         if f1 > best_dev_fb1:
    #             best_dev_fb1, best_threshold, best_algorithm = f1, threshold, algorithm
    #             best_precision, best_recall = precision, recall

    #             # update threshold
    #             config.threshold = best_threshold
    #             config.customized_threshold = customized

    #             logger.info( '%s\n%s' % ('validation', info) ) 


    # ================================================================================


    # report validation set performance
    source = imap( lambda x: x[1],
                   ifilter( lambda x : x[0] % 10 >= 9,
                   enumerate( imap( lambda x: x[:4], 
                                    LoadED( config.data_path + '/%s-eval-parsed' % config.language ) ) ) ) )
    if args.iflytek:
        source = chain( source, 
                        imap( lambda x: x[1],
                              ifilter( lambda x : 90 <= x[0] % 100 < 95,
                                       enumerate( imap( lambda x: x[:4], 
                                                  LoadED( 'iflytek-clean-%s' % config.language ) ) ) ) ) )

    precision, recall, f1, info = evaluation( PredictionParser( source,
                                                                valid_file, 
                                                                config.n_window,
                                                                n_label_type = config.n_label_type ), 
                                              config.threshold, 
                                              config.algorithm, 
                                              True,
                                              analysis = None,
                                              n_label_type = config.n_label_type,
                                              decoder_callback = config.customized_threshold )
    logger.info( '%s\n%s' % ('validation', info) ) 


    if config.language != 'spa':
        # report test set performance
        source = imap( lambda x: x[1],
                       ifilter( lambda x: x[0] % 100 >= 95,
                                enumerate( imap( lambda x: x[:4],
                                           LoadED( 'iflytek-clean-%s' % config.language ) ) ) ) )

        precision, recall, f1, info = evaluation( PredictionParser( source,
                                                                    test_file, 
                                                                    config.n_window,
                                                                    n_label_type = config.n_label_type ), 
                                                  config.threshold, 
                                                  config.algorithm, 
                                                  True,
                                                  analysis = None,
                                                  n_label_type = config.n_label_type,
                                                  decoder_callback = config.customized_threshold )
        logger.info( '%s\n%s' % ('test', info) )


    with open( args.basename + '.config', 'wb' ) as fp:
        cPickle.dump( config, fp )
    logger.info( 'customized threshold is stored in config' )
