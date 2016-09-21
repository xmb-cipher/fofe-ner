#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import numpy, os, codecs, itertools, logging
from gigaword2feature import *
from scipy.sparse import csr_matrix
from sklearn import preprocessing

logger = logging.getLogger( __name__ )


def LoadED( rspecifier, language = 'eng' ):

    entity2cls = {  # KBP2015 label
                    'PER_NAM' : 0, 
                    'PER_NOM' : 5, 
                    'ORG_NAM' : 1, 
                    'GPE_NAM' : 2, 
                    'LOC_NAM' : 3, 
                    'FAC_NAM' : 4, 
                    'TTL_NAM' : 5,

                    # iflytek label
                    'PER_NAME' : 0,  
                    'ORG_NAME' : 1, 
                    'GPE_NAME' : 2, 
                    'LOC_NAME' : 3, 
                    'FAC_NAME' : 4, 
                    'PER_NOMINAL' : 5,
                    'ORG_NOMINAL' : 6,
                    'GPE_NOMINAL' : 7,
                    'LOC_NOMINAL' : 8,
                    'FAC_NOMINAL' : 9,
                    'TITLE_NAME' : 5,
                    'TITLE_NOMINAL' : 5
                } 

    if os.path.isfile( rspecifier ):
        with codecs.open( rspecifier, 'rb', 'utf8' ) as fp:
            processed, original = fp.read().split( u'=' * 128, 1 )
            original = original.strip()

            # texts, tags, failures = processed.split( u'\n\n\n', 2 )
            texts = processed.split( u'\n\n\n' )[0]
            for text in texts.split( u'\n\n' ):
                parts = text.split( u'\n' )
                # assert len(parts) in [2, 3], 'sentence, offsets, labels(optional)'
                if len( parts ) not in [2, 3]:
                    logger.exception( text )
                    continue

                sent, boe, eoe, target, mids, spelling = parts[0].split(u' '), [], [], [], [], []
                offsets = map( lambda x : (int(x[0]), int(x[1])),
                               [ offsets[1:-1].split(u',') for offsets in parts[1].split() ] )
                assert len(offsets) == len(sent), rspecifier + '\n' + \
                        str( offsets ) + '\n' + str( sent ) + '\n%d vs %d' % (len(offsets), len(sent))

                if len(parts) == 3:
                    for ans in parts[-1].split():
                        try:
                            begin_idx, end_idx, mid, mention1, mention2 = ans[1:-1].split(u',')
                            target.append( entity2cls[str(mention1 + u'_' + mention2)] )
                            boe.append( int(begin_idx) )
                            eoe.append( int(end_idx) )
                            mids.append( mid )
                            spelling.append( original[ offsets[boe[-1]][0] : offsets[eoe[-1] - 1][1] ] )
                        except ValueError as ex1:
                            logger.exception( rspecifier )
                            logger.exception( ans )
                        except KeyError as ex2:
                            logger.exception( rspecifier )
                            logger.exception( ans )

                        try:
                            assert 0 <= boe[-1] < eoe[-1] <= len(sent), \
                                    '%s  %d  ' % (rspecifier.split('/')[-1], len(sent)) + \
                                    '  '.join( str(x) for x in [sent, boe, eoe, target, mids] )
                        except IndexError as ex:
                            logger.exception( rspecifier )
                            logger.exception( str(boe) + '   ' + str(eoe) )
                            continue
                    assert( len(boe) == len(eoe) == len(target) == len(mids) )

                # move this part to processed_sentence
                # if language == 'eng':
                #     for i,w in enumerate( sent ):
                #         sent[i] = u''.join( c if 0 <= ord(c) < 128 else chr(0) for c in list(w) )
                yield sent, boe, eoe, target, mids, spelling


    else:
        for filename in os.listdir( rspecifier ):
            for X in LoadED( os.path.join( rspecifier, filename ), language ):
                yield X



def LoadEL( rspecifier, language = 'eng', window = 1 ):
    if os.path.isfile( rspecifier ):
        data = list( LoadED( rspecifier, language ) )
        for i,(sent,boe,eoe,label,mid,spelling) in enumerate(data):
            if len(label) > 0:
                previous, next = [], []
                for s,_,_,_,_,_ in data[i - window: i]:
                    previous.extend( s )
                for s,_,_,_,_,_ in data[i + 1: i + 1 + window]:
                    next.extend( s )
                yield previous + sent + next, \
                      [ len(previous) + b for b in boe ], \
                      [ len(previous) + e for e in eoe ], \
                      label, mid, spelling

    else:
        for filename in os.listdir( rspecifier ):
            for X in LoadEL( os.path.join( rspecifier, filename ), language ):
                yield X



def PositiveEL( embedding_basename,
                rspecifier, language = 'eng', window = 1 ):

    raw_data = list( LoadEL( rspecifier, language, window ) )

    # with open( embedding_basename + '.word2vec', 'rb' ) as fp:
    #   shape = numpy.fromfile( fp, dtype = numpy.int32, count = 2 )
    #   projection = numpy.fromfile( fp, dtype = numpy.float32 ).reshape( shape )
    # logger.debug( 'embedding loaded' )

    with codecs.open( embedding_basename + '.wordlist', 'rb', 'utf8' ) as fp:
        n_word = len( fp.read().strip().split() )
    logger.debug( 'a vocabulary of %d words is used' % n_word )

    numericizer = vocabulary( embedding_basename + '.wordlist', 
                              case_sensitive = False )

    bc = batch_constructor( [ rd[:4] for rd in raw_data ],
                              numericizer, numericizer, 
                              window = 1024, n_label_type = 7 )
    logger.debug( bc )

    index_filter = set([2, 3, 6, 7, 8])

    mid_itr = itertools.chain.from_iterable( rd[-2] for rd in raw_data )

    mention = itertools.chain.from_iterable( rd[-1] for rd in raw_data )

    # for sent, boe, eoe, _, _ in raw_data: 
    #   for b,e in zip( boe, eoe ):
    #       mention.append( sent[b:e] )

    # feature_itr = bc.mini_batch( 1, 
    #                            shuffle_needed = False, 
    #                            overlap_rate = 0, disjoint_rate = 0, 
    #                            feature_choice = 7  )
    # # assert( len(list(mid_itr)) == len(list(feature_itr)) )

    # for mid, feature in zip( mid_itr, feature_itr ):
    #   yield mid, \
    #         [ f.reshape([-1])[1::2] if i in index_filter else f.reshape([-1]) \
    #           for i,f in enumerate(feature[:9]) ]

    l1v, r1v, l1i, r1i, l2v, r2v, l2i, r2i, bow = \
            bc.mini_batch( len(bc.positive), 
                           shuffle_needed = False, 
                           overlap_rate = 0, 
                           disjoint_rate = 0, 
                           feature_choice = 7  ).next()[:9]
    l1 = csr_matrix( ( l1v, ( l1i[:,0].reshape([-1]), l1i[:,1].reshape([-1]) ) ),
                     shape = [len(bc.positive), n_word] ).astype( numpy.float32 )
    l2 = csr_matrix( ( l2v, ( l2i[:,0].reshape([-1]), l2i[:,1].reshape([-1]) ) ),
                     shape = [len(bc.positive), n_word] ).astype( numpy.float32 )
    r1 = csr_matrix( ( r1v, ( r1i[:,0].reshape([-1]), r1i[:,1].reshape([-1]) ) ),
                     shape = [len(bc.positive), n_word] ).astype( numpy.float32 )
    r2 = csr_matrix( ( r2v, ( r2i[:,0].reshape([-1]), r2i[:,1].reshape([-1]) ) ),
                     shape = [len(bc.positive), n_word] ).astype( numpy.float32 )
    bow = csr_matrix( ( numpy.ones( bow.shape[0] ),
                        ( bow[:,0].reshape([-1]), bow[:,1].reshape([-1]) ) ),
                      shape = [len(bc.positive), n_word] ).astype( numpy.float32 )
    return list(mid_itr), mention, l1, l2, r1, r2, bow

    


def LoadTfidf( tfidf_basename, col ):
    indices = numpy.fromfile( tfidf_basename + '.indices', dtype = numpy.int32 )
    data = numpy.fromfile( tfidf_basename + '.data', dtype = numpy.float32 )
    indptr = numpy.fromfile( tfidf_basename + '.indptr', dtype = numpy.int32 )
    assert indices.shape == data.shape

    mid2tfidf = csr_matrix( (data, indices, indptr), 
                            shape = (indptr.shape[0] - 1, col) )
    del data, indices, indptr
    mid2tfidf = mid2tfidf.astype( numpy.float32 )

    with open( tfidf_basename + '.list' ) as fp:
        idx2mid = [ mid[1:-1] for mid in fp.readlines() ]
        mid2idx = { m:i for i,m in enumerate( idx2mid ) }

    return mid2tfidf, idx2mid, mid2idx



if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level = logging.DEBUG )

    embedding_basename = 'word2vec/gigaword128-case-insensitive'
    tfidf_basename = '/eecs/research/asr/Shared/Entity_Linking_training_data_from_Freebase/mid2tfidf'

    with open( embedding_basename + '.word2vec', 'rb' ) as fp:
        shape = numpy.fromfile( fp, dtype = numpy.int32, count = 2 )
        projection = numpy.fromfile( fp, dtype = numpy.float32 ).reshape( shape )
    logger.info( 'embedding loaded' )

    solution, mention, l1, l2, r1, r2, bow = PositiveEL( embedding_basename,
                                                         'kbp-raw-data/eng-train-parsed' )
    logger.info( 'fofe loaded' )

    mid2tfidf, idx2mid, mid2idx = LoadTfidf( tfidf_basename, projection.shape[0] )
    logger.info( 'tfidf loaded' )

    l1p = l1.dot( projection )
    l2p = l2.dot( projection )
    r1p = r1.dot( projection )
    r2p = r2.dot( projection )
    bowp = bow.dot( projection )
    mid2tfidfp = mid2tfidf.dot( projection )
    logger.info( 'projection done' )
    del l1, l2, r1, r2, bow, mid2tfidf

    bow_coef = 0.5

    feature = bow_coef * bowp + (1. - bowp) * (l2p + r2p) / 2.
    del l1p, l2p, r1p, r2p, bowp

    normalized_feature = preprocessing.normalize(feature, norm = 'l2')
    logger.info( 'feature computed & normalized' )
    del feature

    normalized_mid2tfidfp = preprocessing.normalize(mid2tfidfp, norm = 'l2')
    logger.info( 'tfidf normalized' )
    del mid2tfidfp


    for i,(s,m) in enumerate( zip( solution, mention ) ):
        print s, m
        # similarity = numpy.dot( normalized_feature[i:i + 1], normalized_mid2tfidfp.T )
        # top = numpy.argsort( similarity, axis = 1, kind = 'heapsort' )
        # print m, s, idx2mid[top[0,-1]]
