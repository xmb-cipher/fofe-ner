#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import numpy, argparse, logging, time, cPickle, codecs


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
    parser.add_argument( '--buffer', type = str, default = 'eval-buffer' )

    args = parser.parse_args()
    logger.info( str(args) + '\n' )

    from fofe_mention_net import *

    with open( args.basename + '.config', 'rb' ) as fp:
        config = cPickle.load( fp )
    logger.info( config.__dict__ )
    logger.info( 'configuration loaded' )

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

    # kbp_gazetteer = gazetteer( config.data_path + '/kbp-gazetteer' )
    kbp_gazetteer = gazetteer( config.data_path + '/eng-gaz', mode = 'KBP' )

    # idx2ner = [ 'PER_NAM', 'PER_NOM', 'ORG_NAM', 'GPE_NAM', 'LOC_NAM', 'FAC_NAM', 'TTL_NAM', 'O'  ]
    idx2ner = [ 'PER_NAM', 'ORG_NAM', 'GPE_NAM', 'LOC_NAM', 'FAC_NAM',
                'PER_NOM', 'ORG_NOM', 'GPE_NOM', 'LOC_NOM', 'FAC_NOM',
                'O' ]  

    # ==========================================================================================
    # go through validation set to see if we load the correct file

    # load 10% KBP test data
    def __LoadValidation():
        source = imap( lambda x: x[1],
                       ifilter( lambda x : x[0] % 10 >= 9,
                       enumerate( imap( lambda x: x[:4], 
                                        LoadED( config.data_path + '/%s-eval-parsed' % config.language ) ) ) ) )
        # load 5% iflytek data
        if config.language != 'spa':
            source = chain( source, 
                            imap( lambda x: x[1],
                                  ifilter( lambda x : 90 <= x[0] % 100 < 95,
                                           enumerate( imap( lambda x: x[:4], 
                                                      LoadED( 'iflytek-clean-%s' % config.language ) ) ) ) ) )
        return source

    # istantiate a batch constructor
    valid = batch_constructor( __LoadValidation(),
                               numericizer1, numericizer2, gazetteer = kbp_gazetteer, 
                               alpha = config.word_alpha, window = config.n_window, 
                               n_label_type = config.n_label_type,
                               language = config.language )

    logger.info( 'valid: ' + str(valid) )


    # go through validation set
    with open( args.buffer, 'wb' ) as valid_predicted:
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


    pp = [ p for p in PredictionParser( __LoadValidation(),
                                        args.buffer, 
                                        config.n_window,
                                        n_label_type = config.n_label_type ) ]

    _, _, best_dev_fb1, info = evaluation( pp, config.threshold, config.algorithm, True, 
                                            n_label_type = config.n_label_type,
                                            decoder_callback = config.customized_threshold )
    logger.info( '%s\n%s' % ('validation', info) ) 

    # ==========================================================================================

    for filename in os.listdir( args.in_dir ):
        full_name = os.path.join( args.in_dir, filename )

        with codecs.open( full_name, 'rb', 'utf8' ) as fp:
            logger.info( '*' * 32 + '  ' + filename + '  ' + '*'* 32 + '\n' )

            processed, original = fp.read().split( u'=' * 128, 1 )
            original = original.strip()

            texts, tags, failures = processed.split( u'\n\n\n', 2 )
            texts = [ text.split( u'\n' ) for text in texts.split( u'\n\n' ) ]
            
            data = batch_constructor( # imap( lambda x: (x[0].split(u' '), [], [], []), texts ), 
                                      imap( lambda x: x[:4], LoadED( full_name ) ),
                                      numericizer1, numericizer2, gazetteer = kbp_gazetteer, 
                                      alpha = config.word_alpha, window = config.n_window, 
                                      n_label_type = config.n_label_type,
                                      language = config.language )
            logger.info( 'data: ' + str(data) )

            with open( args.buffer, 'wb' ) as buff_file:
                for example in data.mini_batch_multi_thread( 512, False, 1, 1, config.feature_choice ):
                    _, pi, pv = mention_net.eval( example )

                    # expcted has gargadge values
                    for estimate, probability in zip( pi, pv ):
                        print >> buff_file, '%d  %d  %s' % \
                                (-1, estimate, '  '.join( [('%f' % x) for x in probability.tolist()] ))

            labeled_text = []
            for (sent, offsets), (s, table, estimate, actual) in zip( texts,
                    PredictionParser( # imap( lambda x: (x[0].split(u' '), [], [], []), texts ),
                                      imap( lambda x: x[:4], LoadED( full_name ) ),
                                      args.buffer, config.n_window,
                                      n_label_type = config.n_label_type ) ):

                estimate = decode( s, estimate, table, config.threshold, config.algorithm,
                                   config.customized_threshold ) 
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


