#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import numpy, argparse, logging, time, cPickle, codecs, os, getpass, sys
from subprocess import Popen, PIPE, call

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
    parser.add_argument( '--nfold1st', action = 'store_true', default = False,
                         help = 'load 5 models for 1st pass if set' )
    parser.add_argument( '--nfold2nd', action = 'store_true', default = False,
                         help = 'load 5 models for 2nd pass if set' )

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

    if args.nfold1st:
        model1st = [ ('%s-%d' % (args.model1st, i)) for i in xrange(5) ]
        logger.info( 'The evaluator will load 5 models' )
    else:
        model1st = [ args.model1st ]
        logger.info( 'The evaluator will load a single model' )
    algo2freq, threshold1, prob1 = { 1: 0, 2: 0, 3: 0 }, 0, None

    for model in model1st:

        with open( '%s.config' % model, 'rb' ) as fp:
            config1 = cPickle.load( fp )
        config1.l1 = 0
        config1.l2 = 0
        logger.info( config1.__dict__ )
        logger.info( 'config1st loaded' )

        algo2freq[config1.algorithm] += 1
        threshold1 += config1.threshold / len(model1st)

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
        mention_net.fromfile( model )
        logger.info( 'model loaded' )

        ################################################################################

        print1 = []
        for example in test.mini_batch_multi_thread( 
                            2560 if config1.feature_choice & (1 << 9) > 0 else 2560, 
                            False, 1, 1, config1.feature_choice ):

            _, pi, pv = mention_net.eval( example )
            print1.append( numpy.concatenate( 
                               ( example[-1].astype(numpy.float32).reshape(-1, 1),
                                 pi.astype(numpy.float32).reshape(-1, 1),
                                 pv ), axis = 1 ) )

        del mention_net
        print1 = numpy.concatenate( print1, axis = 0 )
        logger.info( 'probability evaluated for %s' % model )

        if prob1 is None:
            prob1 = print1
        else:
            prob1 += print1

    if len(model1st) > 1:
        prob1 /= len(model1st)
        prob1[:,1:2] = prob1[:,2:].argmax( axis = 1 ).reshape(-1, 1)
    predicted1st = os.path.join( buf_dir, 'predict1st' )
    numpy.savetxt( predicted1st, prob1, 
                   fmt = '%d  %d' + '  %f' * (config1.n_label_type + 1) )

    logger.info( 'evaluation set passed first time' )

    pp = list(PredictionParser( SampleGenerator( args.testb ), predicted1st, config1.n_window ))

    for threshold in numpy.arange(0.1, 1, 0.1):
        output1st = os.path.join(buf_dir, 'output1st')
        with open( output1st, 'wb' ) as out1st:
            algorithm1 = sorted([(y, x) for (x, y) in algo2freq.items()], reverse = True)[0][1]     
            _, _, _, info = evaluation( pp, threshold, algorithm1,
                                        conll2003out = out1st,
                                        sentence_iterator = SentenceIterator( args.testb ) )
        logger.info( '\n' + info )

        cmd = 'cat %s | conlleval' % output1st
        process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
        (out, err) = process.communicate()
        exit_code = process.wait()
        logger.info( '\x1B[32mofficial\n%s\x1B[0m' % out )

    logger.info( 'first-round output generated (threshold %d, algorithm %d)' \
                    % (threshold1, algorithm1) )

    ################################################################################
    ########## compute 2nd-pass result
    ################################################################################

    if args.nfold2nd:
        model2nd = [ ('%s-%d' % (args.model2nd, i)) for i in xrange(5) ]
        logger.info( 'The evaluator will load 5 models' )
    else:
        model2nd = [ args.model2nd ]
        logger.info( 'The evaluator will load a single model' )
    algo2freq, threshold2, prob2 = { 1: 0, 2: 0, 3: 0 }, 0, None

    for model in model2nd:

        with open( '%s.config' % model, 'rb' ) as fp:
            config2 = cPickle.load( fp )
        config2.l1 = 0
        config2.l2 = 0
        config2.is_2nd_pass = True
        assert config2.n_window == config1.n_window, 'inconsisitent window size'
        logger.info( config2.__dict__ )
        logger.info( 'config2nd loaded' )

        algo2freq[config2.algorithm] += 1
        threshold2 += config2.threshold / len(model2nd)

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
        mention_net.fromfile( model )
        logger.info( 'model loaded' )

        ################################################################################

        print2 = []
        for example in test.mini_batch_multi_thread( 
                            2560 if config1.feature_choice & (1 << 9) > 0 else 2560, 
                            False, 1, 1, config1.feature_choice ):

            _, pi, pv = mention_net.eval( example )
            print2.append( numpy.concatenate( 
                               ( example[-1].astype(numpy.float32).reshape(-1, 1),
                                 pi.astype(numpy.float32).reshape(-1, 1),
                                 pv ), axis = 1 ) )
        
        del mention_net
        print2 = numpy.concatenate( print2, axis = 0 )
        logger.info( 'probability evaluated for %s' % model )

        if prob2 is None:
            prob2 = print2
        else:
            prob2 += print2

    if len(model2nd) > 1:
        prob2 /= len(model2nd)
        prob2[:,1:2] = prob2[:,2:].argmax( axis = 1 ).reshape(-1, 1)

    predicted2nd = os.path.join( buf_dir, 'predict2nd' )
    numpy.savetxt( predicted2nd, prob2, 
                   fmt = '%d  %d' + '  %f' * (config2.n_label_type + 1) )

    logger.info( 'evaluation set passed second time' )

    pp = PredictionParser( SampleGenerator( args.testb ), predicted2nd, config2.n_window )

    output2nd = os.path.join(buf_dir, 'output2nd')
    with open( output2nd, 'wb' ) as out2nd:
        algorithm2 = sorted([(y, x) for (x, y) in algo2freq.items()], reverse = True)[0][1]
        _, _, _, info = evaluation( pp, threshold2, algorithm2,
                                    conll2003out = out2nd,
                                    sentence_iterator = SentenceIterator( args.testb ) )
    # logger.info( 'non-official\n' + info )

    cmd = 'cat %s | conlleval' % output2nd
    process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
    (out, err) = process.communicate()
    exit_code = process.wait()
    logger.info( 'official\n' + out )

    logger.info( 'second-round output generated' )

    ################################################################################
    ########## combine 1st-pass and 2nd-pass result in terms of raw probability
    ################################################################################

    # for weight in numpy.arange(0.1, 1, 0.1):
    for weight in [0.618]:
        prob3 = weight * prob1 + (1 - weight) * prob2

        prob3[:,1:2] = prob3[:,2:].argmax( axis = 1 ).reshape(-1, 1)
        threshold = (threshold1 + threshold2) / 2
        logger.info( '\x1B[32mthreshold1: %f, threshold2: %f\n\x1B[0m' % (threshold1, threshold2) )

        predicted3rd = os.path.join( buf_dir, 'predict3rd' )
        numpy.savetxt( predicted3rd, prob3, 
                       fmt = '%d  %d' + '  %f' * (config2.n_label_type + 1) )

        pp = list( PredictionParser( SampleGenerator( args.testb ), predicted3rd, config2.n_window ) )

        # for threshold in numpy.arange(0.2, 1, 0.1):
        for threshold in [0.4]:
            output3rd = os.path.join(buf_dir, 'output3rd')
            with open( output3rd, 'wb' ) as out3rd:
                _, _, _, info = evaluation( pp, threshold, config2.algorithm,
                                            conll2003out = out3rd,
                                            sentence_iterator = SentenceIterator( args.testb ) )
            # logger.info( 'non-official\n' + info )
                
            cmd = 'cat %s | conlleval' % output3rd
            process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
            (out, err) = process.communicate()
            exit_code = process.wait()
            logger.info( '%f * prob1 + %f * prob2' % (weight, (1 - weight)) )
            logger.info( 'threshold: %f' % threshold )
            logger.info( '\x1B[32mofficial\n%s\x1B[0m' % out )

            cmd = 'visualizer/compose-html.py %s %s %s; cp %s visualizer/error.testb' % \
                    (args.testb, output3rd, 'visualizer/error.html', output3rd)
            process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
            (out, err) = process.communicate()
            exit_code = process.wait()

    logger.info( '1st-pass and 2nd-pass combined at probability level' )
