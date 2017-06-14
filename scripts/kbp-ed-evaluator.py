#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import numpy, argparse, logging, time, cPickle, codecs
from io import BytesIO


# set a logging file at DEBUG level, TODO: windows doesn't allow ":" appear in a file name
logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                     level= logging.DEBUG,
                     filename = ('log/kbp ' + time.ctime() + '.log').replace(' ', '-'), 
                     filemode = 'w' )

# direct the INFO-level logging to the screen
console = logging.StreamHandler()
console.setLevel( logging.INFO )
console.setFormatter( logging.Formatter( '%(asctime)s : %(levelname)s : %(message)s' ) )
logging.getLogger().addHandler( console )

logger = logging.getLogger( __name__ )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( 'basename', type = str )
    parser.add_argument( 'in_dir', type = str )
    parser.add_argument( 'out_dir', type = str )
    parser.add_argument( '--nfold', action = 'store_true', default = False )

    args = parser.parse_args()
    logger.info( str(args) + '\n' )

    from fofe_mention_net import *

    threshold = numpy.zeros( (2,), dtype = numpy.float32 )
    algorithm = {}
    config_list, mention_net_list = [], []
    basename_list = [ args.basename ]
    if args.nfold:
        basename_list = [ '%s-%d' % (args.basename, i) for i in xrange(5) ]

    for basename in basename_list:
        config = mention_config()
        with open( basename + '.config', 'rb' ) as fp:
            config.__dict__.update( cPickle.load( fp ).__dict__ )
        logger.info( config.__dict__ )
        config_list.append( config )

        threshold += numpy.asarray( config.threshold, dtype = numpy.float32 )
        config.algorithm = tuple(config.algorithm)
        if config.algorithm in algorithm:
            algorithm[config.algorithm] += 1
        else:
            algorithm[config.algorithm] = 1

        if config.version == 2:
            mention_net = fofe_mention_net_v2( config )
        else:
            mention_net = fofe_mention_net( config )
        mention_net.fromfile( basename )
        mention_net_list.append( mention_net )

    if config.language != 'cmn':
        numericizer1 = vocabulary( args.basename + '-case-insensitive.wordlist', 
                                   config.char_alpha, False )
        numericizer2 = vocabulary( args.basename + '-case-sensitive.wordlist', 
                                   config.char_alpha, True )
    else:
        numericizer1 = chinese_word_vocab( args.basename + '-char.wordlist' )
        numericizer2 = chinese_word_vocab( args.basename + \
                            ('-avg.wordlist' if config.average else '-word.wordlist') )

    logger.info( 'config, model & vocab loaded' )

    try:
        pkl_path = "%s.pkl" % args.basename
        with open( pkl_path, 'rb' ) as fp:
            kbp_gazetteer = cPickle.load( fp )
    except:
        txt_path = "%s.gaz" % args.basename
        kbp_gazetteer = gazetteer( txt_path, mode = 'KBP' )

    idx2ner = [ 'PER_NAM', 'ORG_NAM', 'GPE_NAM', 'LOC_NAM', 'FAC_NAM',
                'PER_NOM', 'ORG_NOM', 'GPE_NOM', 'LOC_NOM', 'FAC_NOM',
                'O' ]

    if args.nfold:
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


    # ==========================================================================================

    for filename in os.listdir( args.in_dir ):
        full_name = os.path.join( args.in_dir, filename )

        with codecs.open( full_name, 'rb', 'utf8' ) as fp:
            logger.info( '*' * 32 + '  ' + filename + '  ' + '*'* 32 + '\n' )

            processed, original = fp.read().split( u'=' * 128, 1 )
            original = original.strip()

            texts, tags, failures = processed.split( u'\n\n\n', 2 )
            texts = [ text.split( u'\n' ) for text in texts.split( u'\n\n' ) ]
            
            target_func = lambda x: x['target'] if config.version == 2 else x[-1]

            for i, mention_net in enumerate( mention_net_list ):
                if config.version == 2:
                    data = batch_constructor_v2( 
                        imap( lambda x: x[:4], LoadED( full_name ) ),
                        numericizer1, 
                        numericizer2, 
                        gazetteer = kbp_gazetteer,  
                        window = config.n_window, 
                        n_label_type = config.n_label_type,
                        language = config.language 
                    )
                else:
                    data = batch_constructor( 
                        imap( lambda x: x[:4], LoadED( full_name ) ),
                        numericizer1, 
                        numericizer2, 
                        gazetteer = kbp_gazetteer, 
                        alpha = config.word_alpha, 
                        window = config.n_window, 
                        n_label_type = config.n_label_type,
                        language = config.language 
                    )
                logger.info( 'data: ' + str(data) )

                prob = []
                for example in data.mini_batch_multi_thread( 512, False, 1, 1, config.feature_choice ):
                    _, pi, pv = mention_net.eval( example )

                    prob.append(
                        numpy.concatenate(
                            ( target_func(example).astype(numpy.float32).reshape(-1, 1),
                              pi.astype(numpy.float32).reshape(-1, 1),
                              pv ),
                            axis = 1
                        )
                    )
                prob = numpy.concatenate( prob, axis = 0 )

                if i == 0:
                    prob_avg = prob
                else:
                    prob_avg += prob

            prob_avg /= len(mention_net_list)
            prob_avg[:,1] = prob_avg[:,2:].argmax( axis = 1 )

            memory = BytesIO()
            numpy.savetxt( 
                memory, 
                prob_avg, 
                fmt = '%d  %d' + '  %f' * (config.n_label_type + 1) 
            )
            memory.seek(0)

            labeled_text = []
            for (sent, offsets), (s, table, estimate, actual) in zip( texts,
                    PredictionParser( imap( lambda x: x[:4], LoadED( full_name ) ),
                                      memory, config.n_window,
                                      n_label_type = config.n_label_type ) ):

                estimate = decode( s, estimate, table, threshold, algorithm ) 
                estimate = [ u'(%d,%d,DUMMY,%s,%s)' % \
                             tuple([b,e] + idx2ner[c].split('_')) for b,e,c in sorted(estimate) ]
                span = sent + u'\n' + offsets
                if len( estimate ) > 0:
                    span += u'\n' + u' '.join( estimate )
                labeled_text.append( span ) 


        full_name = os.path.join( args.out_dir, filename )
        labeled_text = u'\n\n'.join( labeled_text )

        with codecs.open( full_name, 'wb', 'utf8' ) as fp:
            fp.write( u'\n\n\n'.join( [labeled_text, tags, failures] ) )
            fp.write( u'\n\n' + u'=' * 128 + u'\n\n\n' + original )

        logger.info( '%s processed\n' % filename )


