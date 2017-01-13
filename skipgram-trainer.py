#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

"""
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : skipgram-trainer.py
Last Update : Mar 25, 2016
Description : A wrapper of skip-gram from gensim
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

History
    20160325 regex added to handle general number, date and phone number

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
"""


import numpy
import gensim
import multiprocessing
import argparse
import os
import re
import os
import codecs
import logging
from hanziconv import HanziConv
logger = logging.getLogger()


__date1 = re.compile(
    r"^(?:(?:31(\/|-|\.)(?:0?[13578]|1[02]|(?:Jan|Mar|May|Jul|Aug|Oct|Dec)))\1|(?:(?:29|30)(\/|-|\.)(?:0?[1,3-9]|1[0-2]|(?:Jan|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.)(?:0?2|(?:Feb))\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.)(?:(?:0?[1-9]|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep))|(?:1[0-2]|(?:Oct|Nov|Dec)))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$".encode('utf8')
)

__date2 = re.compile(
    r"^(((\d{4}(\/|-|\.)((0[13578](\/|-|\.)|1[02](\/|-|\.))(0[1-9]|[12]\d|3[01])|(0[13456789](\/|-|\.)|1[012](\/|-|\.))(0[1-9]|[12]\d|30)|02(\/|-|\.)(0[1-9]|1\d|2[0-8])))|((([02468][048]|[13579][26])00|\d{2}([13579][26]|0[48]|[2468][048])))(\/|-|\.)02(\/|-|\.)29)){0,10}$".encode('utf8')
)

__number = re.compile(
    r"^(\+|-)?(([1-9]\d{0,2}(,\d{3})*)|([1-9]\d*)|0)(\.\d+)?$".encode('utf8')
)

__phone = re.compile(
    r"^(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?$".encode('utf8')
)

__time = re.compile(
    r"^(?:(?:([01]?\d|2[0-3]):)?([0-5]?\d):)?([0-5]?\d)$".encode('utf8')
)


__has_digit_but_no_letter = re.compile( r"^(?=[^A-Za-z]+$).*[0-9].*$".encode('utf8') )

__has_digit = re.compile( r"^.*[0-9].*$".encode('utf8') )



def sentence_eng( filename, case_sensitive = False ):
    """
    Parameters
    ----------
        filename : str
            directory or filename of the corpus, one sentence per line

    Yields
    ------
        sentence: list of str
            a sentence splited as a list of words
    """ 

    if os.path.isdir( filename ):
        for f in os.listdir( filename ):
            if os.path.isfile( filename + '/' + f ):
                for s in sentence_eng( filename + '/' + f, case_sensitive ):
                    yield s
    else:
        logger.info( 'start to read ' + filename )
        # with codecs.open( filename, 'rb', 'utf8' ) as fp:
        with open( filename, 'rb' ) as fp:
            for line in fp:
                line = line.decode('utf-8','ignore').encode('utf-8')
                if not case_sensitive:
                    s = line.strip().lower().split()
                else:
                    s = line.strip().split()
                for i in xrange(len(s)):
                    if re.match( __has_digit, s[i] ):
                        if re.match( __number, s[i] ):
                            s[i] = u'<numeric-value>'
                        elif re.match( __date1, s[i] ) or re.match( __date2, s[i] ):
                            s[i] = u'<date-value>'
                        elif re.match( __time, s[i] ):
                            s[i] = u'<time-value>'
                        elif re.match( __phone, s[i] ):
                            s[i] = u'<phone-value>'
                        # elif re.match( __digit_without_letter, s[i] ):
                        #     s[i] = '<digit_without_letter>'
                        else:
                            s[i] = u'<contains-digit>'
                yield s



def sentence_cmn( rspecifier, word_level = True ):
    if os.path.isdir( rspecifier ):
        for f in os.listdir( rspecifier ):
            full_name = os.path.join( rspecifier, f )
            for s in sentence_cmn( full_name, word_level ):
                yield s

    else:
        with codecs.open( rspecifier, 'rb', 'utf8' ) as fp:
            for line in fp:
                line = HanziConv.toSimplified( line.strip() )
                if word_level:
                    sent = line.split()
                else:
                    sent = []
                    for w in line.split():
                        has_chinese = any( u'\u4e00' <= c <= u'\u9fff' for c in w )
                        if has_chinese:
                            sent.extend( list(w) )
                        else:
                            sent.append( w )
                
                yield [ u'<numeric>' if re.match(__has_digit_but_no_letter, w) else w for w in sent ]





def skipgram_eng( filename, min_cnt, max_vocab, n_embedding, n_window, 
                  word_list = None, case_sensitive = False ):
    """
    Parameters
    ----------
        filename : str
            filename of the corpus, one sentence per line
        min_cnt : int
            minimum occurance of word to be considered
        max_vocab : int
            maximum number of words to keep
        n_embedding : int
            the dimension of word embedding
        n_windows : int
            the length of context
        case_sensitive : bool
            all words are transformed into lowercase if false

    Returns:
        word_vector : numpy.ndarray
            the word vectors in a 2d matrix; positions are given by 'idx2word'
        idx2word : list
            the vocabulary in sorted order
    """

    n_worker = multiprocessing.cpu_count()
    logger.info( "This machine has %d processors. We'll use %d of them" % 
                (n_worker, n_worker) )

    model = gensim.models.Word2Vec( min_count = min_cnt,
                                    workers = n_worker, 
                                    size = n_embedding, 
                                    window = n_window,
                                    max_vocab_size = max_vocab * 4 if max_vocab is not None else None,
                                    sg = 1,
                                    negative = 7 )                                      
    model.build_vocab( sentence_eng( filename, case_sensitive ) )
    for _ in xrange( 10 ):
        model.train( sentence_eng( filename, case_sensitive ) )

    if os.path.isfile( 'questions-words.txt' ):
        model.accuracy( 'questions-words.txt' )

    if word_list is None:
        idx2word = [ w for w in model.index2word if w != u'<unk>' and w != u'<UNK>'  ]
        if case_sensitive:
            if max_vocab is not None:
                idx2word = idx2word[:max_vocab - 2]
            idx2word.append( u'<UNK>' )
        if max_vocab is not None:
            idx2word = idx2word[:max_vocab - 1]
        idx2word.append( u'<unk>' )
    else:
        with codecs.open( word_list, 'rb', 'utf8' ) as fp:
            idx2word = [ line.strip().split()[0] for line in fp ]

    if not case_sensitive:
        word_vector = numpy.ndarray( (len(idx2word), model.layer1_size), numpy.float32 )
        for i in xrange( len(idx2word) - 1 ):
            word_vector[i] = model[ idx2word[i] ]
        word_vector[ len(idx2word) - 1 ] = word_vector[: len(idx2word) - 1].mean(0)
    else:
        n_lowercase_only = 0
        word_vector = numpy.zeros( (len(idx2word), model.layer1_size), numpy.float32 )
        for i in xrange( len(idx2word) - 2 ):
            word_vector[i] = model[ idx2word[i] ]
            if idx2word[i].islower() == idx2word[i]:
                n_lowercase_only += 1
                word_vector[-1] += word_vector[i]
            else:
                word_vector[-2] += word_vector[i]
        if n_lowercase_only > 0:
            word_vector[-1] = word_vector[-1] / n_lowercase_only
        if len(idx2word) - 2 - n_lowercase_only > 0:
            word_vector[-2] = word_vector[-2] / (len(idx2word) - 2 - n_lowercase_only)

    #model.save( 'skipgram-last-run' )
    return word_vector, idx2word




def skipgram_cmn( filename, min_cnt, max_vocab, n_embedding, n_window, 
                  word_list = None, word_level = True ):

    n_worker = multiprocessing.cpu_count()
    logger.info( "This machine has %d processors. We'll use %d of them" % 
                (n_worker, n_worker) )

    model = gensim.models.Word2Vec( min_count = min_cnt,
                                    workers = n_worker, 
                                    size = n_embedding, 
                                    window = n_window,
                                    max_vocab_size = max_vocab * 3 if max_vocab is not None else None,
                                    sg = 1,
                                    negative = 7 )                                      
    model.build_vocab( sentence_cmn( filename, word_level ) )
    for _ in xrange( 7 ):
        model.train( sentence_cmn( filename, word_level ) )

    if word_list is None:
        idx2word = [ w for w in model.index2word if w != u'<unk>' ]
        if max_vocab is not None:
            idx2word = idx2word[:max_vocab - 1]
        idx2word.append( u'<unk>' )
    else:
        with codecs.open( word_list, 'rb', 'utf8' ) as fp:
            idx2word = [ line.strip().split()[0] for line in fp ]

    word_vector = numpy.ndarray( (len(idx2word), model.layer1_size), numpy.float32 )
    for i in xrange( len(idx2word) - 1 ):
        word_vector[i] = model[ idx2word[i] ]
    word_vector[ len(idx2word) - 1 ] = word_vector[: len(idx2word) - 1].mean(0)

    return word_vector, idx2word



if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level= logging.INFO)

    parser = argparse.ArgumentParser( description = 'a wrapper of SGNS' )
    parser.add_argument( 'corpus', type = str, help = 'text file or a directory containing text files' )
    parser.add_argument( 'basename', type = str, help = 'basename.{word2vec, wordlist} will be created' )
    parser.add_argument( '--min_cnt', type = int, default = 10 )
    parser.add_argument( '--max_vocab', type = int, default = 100000 )
    parser.add_argument( '--n_word_embedding', type = int, default = 128 )
    parser.add_argument( '--n_window', type = int, default = 7 )
    parser.add_argument( '--word_list', type = str, default = None )
    parser.add_argument( '--case_sensitive', action = 'store_true', default = False )
    parser.add_argument( '--language', type = str, default = 'eng', 
                         choices = [ 'eng', 'cmn', 'spa' ]  )
    parser.add_argument( '--word_level', action = 'store_true', default = False )
    args = parser.parse_args()

    logger.info( args )

    if args.language == 'eng':
        word2vec, idx2word = skipgram_eng( args.corpus,
                                           args.min_cnt, 
                                           args.max_vocab,
                                           args.n_word_embedding, 
                                           args.n_window,
                                           args.word_list,
                                           args.case_sensitive )
    elif args.language == 'cmn':
        word2vec, idx2word = skipgram_cmn( args.corpus,
                                           args.min_cnt, 
                                           args.max_vocab,
                                           args.n_word_embedding, 
                                           args.n_window,
                                           args.word_list,
                                           args.word_level )


    with open( args.basename + '.word2vec', 'wb' ) as fp:
        numpy.int32(word2vec.shape[0]).tofile( fp )
        numpy.int32(word2vec.shape[1]).tofile( fp )
        word2vec.tofile( fp )

    with codecs.open( args.basename + '.wordlist', 'wb', 'utf8' ) as fp:
        fp.write( u'\n'.join( idx2word ) )
