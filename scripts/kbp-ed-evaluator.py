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
    parser.add_argument( '--nfold', action = 'store_true', default = False )

    args = parser.parse_args()
    logger.info( str(args) + '\n' )

    from fofe_mention_net import *

    config = mention_config()
    with open( args.basename + '.config', 'rb' ) as fp:
        config.__dict__.update( cPickle.load( fp ).__dict__ )
    logger.info( config.__dict__ )
    logger.info( 'configuration loaded' )

    mention_net = fofe_mention_net( config )
    mention_net.fromfile( args.basename )
    logger.info( 'model loaded' )

    if config.language != 'cmn':
        numericizer1 = vocabulary( args.basename + '-case-insensitive.wordlist', 
                                   config.char_alpha, False )
        numericizer2 = vocabulary( args.basename + '-case-sensitive.wordlist', 
                                   config.char_alpha, True )
    else:
        numericizer1 = chinese_word_vocab( args.basename + '-char.wordlist' )
        numericizer2 = chinese_word_vocab( args.basename + \
                            ('-avg.wordlist' if config.average else '-word.wordlist') )
    logger.info( 'vocabulary loaded' )

    try:
        pkl_path = os.path.join( args.basename, 'kbp-gaz.pkl' )
        with open( pkl_path, 'rb' ) as fp:
            kbp_gazetteer = cPickle.load( fp )
    except:
        txt_path = os.path.join( args.basename, 'kbp-gaz.txt' )
        kbp_gazetteer = gazetteer( txt_path, mode = 'KBP' )

    idx2ner = [ 'PER_NAM', 'ORG_NAM', 'GPE_NAM', 'LOC_NAM', 'FAC_NAM',
                'PER_NOM', 'ORG_NOM', 'GPE_NOM', 'LOC_NOM', 'FAC_NOM',
                'O' ]  


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


