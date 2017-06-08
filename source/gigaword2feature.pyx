"""
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : gigaword2feature.pyx
Last Update : May 25, 2016
Description : N/A
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
"""

cimport cython, numpy
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map as ordered_map
from cython.operator cimport dereference, preincrement

cdef extern from "<algorithm>" namespace "std" nogil:
    void reverse[Iter] ( Iter first, Iter last ) 

cdef extern from "<regex>" namespace "std" nogil:
    cdef cppclass regex:
        regex( string& s ) except +
    bint regex_match( string& s, regex& r )

from scipy.sparse import csr_matrix
from Queue import Queue
from threading import Thread
from itertools import izip, islice, imap, combinations, chain
from hanziconv import HanziConv
import numpy, re, random, logging, codecs, copy

logger = logging.getLogger()


################################################################################


def KBP2015( filename ):
    """
    Parameters
    ----------
        filename : str
            path to directory containing NER-annotated Gigaword

    Yields
    ------
        sentence  : list of str
            original sentence
        ner_begin : list of int
            start indices of NER, inclusive
        ner_end   : list of int
            end indices of NER, excusive
        ner_label : list of int
            The entity type of sentence[ner_begin[i]:ner_end[i]] is label[i]
    """
    cdef int i, cls, cnt
    cdef vector[vector[int]] buffer_stack
    cdef list entity_begin
    cdef list engity_end
    cdef list entity_label
    buffer_stack.resize( 10 )

    logger.info( 'According to Liu, TTL_NAM are all labeled as PER_NOM.' )
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

    with codecs.open( filename ) as text_file:
        for line in text_file:
            line = line.strip()

            # bar3idx = line.rfind( '|||' )
            # sentence = [ tokens.split('#')[1] for tokens in line[:bar3idx].strip().split() ]
            # label = line[bar3idx + 3:].split()
            sentence, label = line.rsplit( '|||', 1 )
            sentence = [ tokens.split('#')[1] for tokens in sentence.strip().split() ]
            label = label.split()

            entity_begin, entity_end, entity_label = [], [], []

            cnt = 0
            for l in label:
                if l == 'X':
                    cnt += 1
                elif l.startswith( '(' ):
                    buffer_stack[ entity2cls[ l[1:] ] ].push_back( cnt )
                elif l.startswith( ')' ):
                    cls = entity2cls[ l[1:] ]
                    entity_begin.append( buffer_stack[cls].back() )
                    entity_end.append( cnt )
                    entity_label.append( cls )
                    buffer_stack[cls].pop_back() 

            if cnt > 0:
                assert cnt == len(sentence)
                for i in range( buffer_stack.size() ):
                    assert buffer_stack[i].size() == 0
                yield sentence, entity_begin, entity_end, entity_label


################################################################################


def gigaword( filename ):
    """
    Parameters
    ----------
        filename : str
            path to directory containing NER-annotated Gigaword

    Yields
    ------
        sentence  : list of str
            original sentence
        ner_begin : list of int
            start indices of NER, inclusive
        ner_end   : list of int
            end indices of NER, excusive
        ner_label : list of int
            The entity type of sentence[ner_begin[i]:ner_end[i]] is label[i]
    """

    ner2cls = { 'PERSON' : 0, 'LOCATION' : 1, 'ORGANIZATION' : 2, 'MISC' : 3 }
    number = set(['PERCENT', 'MONEY', 'NUMBER'])

    with codecs.open( filename, 'rb' ) as text_file:
        n_discard = 0

        for line in text_file:
            line = line.strip().split()
            if len(line) > 0:
                sentence, ner_begin, ner_end, ner_label, last_ner = [], [], [], [], 4
                malformed = False

                for p in line:
                    slash_idx = p.rfind('/')
                    if slash_idx == 0:
                        malformed = True
                        break

                    word = p[:slash_idx].replace('\\/', '/').replace('\\\\', '\\') 
                    label = p[slash_idx + 1:]

                    ner = ner2cls.get( label, 4 );
                    if ner != last_ner:
                        if last_ner != 4:
                            ner_end.append( len(sentence) )
                        if ner != 4:
                            ner_begin.append( len(sentence) );
                            ner_label.append( ner );
                    last_ner = ner

                    sentence.append( word )

                if not malformed:
                    if len(ner_end) < len(ner_begin):
                        ner_end.append( len(sentence) )
                    assert len(ner_end) == len(ner_begin)
                    if len(ner_label) > 0 or random.choice([True, False]):
                        yield sentence, ner_begin, ner_end, ner_label
                else:
                    n_discard += 1
                    logger.info( '%d sentence(s) discarded' % n_discard )

################################################################################


def gazetteer( filename, mode = 'CoNLL2003' ):
    """
    Parameters
    ----------
        filename : str
        mode : str

    Returns
    -------
        result : list of set
    """
    if mode == 'CoNLL2003':
        logger.info( 'Loading CoNLL2003 4-type gazetteer' )
        result = [ set() for _ in xrange(4) ]
        ner2cls = { 'PER' : 0, 'LOC' : 1, 'ORG' : 2, 'MISC' : 3 }
        # with codecs.open(filename, 'rb', 'utf8') as text_file:
        with open( filename, 'rb' ) as text_file:
            for line in text_file:
                line = line.decode('utf-8','ignore').encode('utf-8')
                tokens = line.strip().split( None, 1 )
                if len(tokens) == 2:
                    result[ ner2cls[tokens[0]] ].add( tokens[1] )
    else:
        logger.info( 'Loading KBP 6-type gazetteer' )
        result = [ set() for _ in xrange(7) ]
        ner2cls = { '<PER>' : 0, '<ORG>' : 1, '<GPE>' : 2, 
                    '<LOC>' : 3, '<FAC>' : 4, '<TTL>' : 5 }
        with codecs.open(filename, 'rb', 'utf8') as text_file:
            for line in text_file:
                tokens = line.strip().rsplit( None, 1 )
                if len(tokens) == 2 and tokens[1] in ner2cls:
                    result[ ner2cls[tokens[1]] ].add( HanziConv.toSimplified(tokens[0][1:-1]) )

    logger.info( '; '.join( str((cls,len(result[ner2cls[cls]]))) for cls in ner2cls ) )
    return result

################################################################################


def CoNLL2003( filename ):
    """
    Parameters
    ----------
        filename : str
            path to one of eng.{train,testa,testb}

    Yields
    ------
        sentence  : list of str
            original sentence
        ner_begin : list of int
            start indices of NER, inclusive
        ner_end   : list of int
            end indices of NER, excusive
        ner_label : list of int
            The entity type of sentence[ner_begin[i]:ner_end[i]] is label[i]
    """
    ner2cls = { 'B-PER' : 0, 'I-PER' : 0,
                'B-LOC' : 1, 'I-LOC' : 1,
                'B-ORG' : 2, 'I-ORG' : 2,
                'B-MISC' : 3, 'I-MISC' : 3 }
    sentence, ner_begin, ner_end, ner_label, last_ner = [], [], [], [], 4

    with codecs.open( filename, 'rb', 'utf8' ) as text_file:
        for line in text_file:
            tokens = line.strip().split()

            if len(tokens) > 1:
                word, label = tokens[0], tokens[-1]
                ner = ner2cls.get( label, 4 );
                if ner != last_ner:
                    if last_ner != 4:
                        ner_end.append( len(sentence) )
                    if ner != 4:
                        ner_begin.append( len(sentence) );
                        ner_label.append( ner );
                last_ner = ner
                sentence.append( word )

            else:
                if len(sentence) > 0:
                    if len(ner_end) < len(ner_begin):
                        ner_end.append( len(sentence) )
                    assert len(ner_end) == len(ner_begin)
                    yield sentence, ner_begin, ner_end, ner_label
                    # if filename.endswith( 'eng.train' ) and 3 in ner_label:
                    #     yield sentence, ner_begin, ner_end, ner_label
                    sentence, ner_begin, ner_end, ner_label, last_ner = [], [], [], [], 4

################################################################################


def prepare_mini_batch( batch_generator, batch_buffer, timeout ):
    """
    Put every single element that 'batch_generator' yields into 'batch_buffer'. 

    batch_generator : iterable
    batch_buffer : Queue
    """
    for x in batch_generator:
        batch_buffer.put( x, True, timeout )
    batch_buffer.put( None, True, timeout )


################################################################################

class chinese_char_vocab( object ):
    def __init__( self, filename, n_label_type = 0 ):
        self.number = re.compile( r"^(?=[^A-Za-z]+$).*[0-9].*$".encode('utf8') )

        with codecs.open( filename, 'rb', 'utf8' ) as fp:
            self.idx2char = [ c.strip() for c in fp.read().strip().split( u'\n' ) ]
        self.char2idx = { c:i for (i,c) in enumerate( self.idx2char ) }

        self.n_char = len(self.char2idx) - n_label_type - 1
        if n_label_type > 0:
            for c in self.char2idx.keys():
                if c != '<unk>' and self.char2idx[c] >= self.n_char:
                    self.char2idx.pop( c, None )



    def __len__( self ):
        return self.n_char


    def sentence2indices( self, sentence ):
        # This must be same as in "skipgram-trainer.py"
        chars, c_unk = [], self.word2idx[u'<unk>']
        for w in sentence:
            has_chinese = any( u'\u4e00' <= c <= u'\u9fff' for c in w )
            if has_chinese:
                chars.extend( list(w) )
            else:
                chars.append( w )

        return [ self.char2idx.get(c, c_unk) for c in \
                 imap( lambda c: u'<numeric>' if re.match(self.number, c) else c, chars ) ]


################################################################################


cdef class vocabulary( object ):
    cdef dict word2idx
    cdef dict word2fofe
    cdef readonly float alpha
    cdef bint case_sensitive
    cdef int n_word
    cdef int pad_idx
    cdef regex* date_pattern_1
    cdef regex* date_pattern_2
    cdef regex* number_pattern
    cdef regex* phone_pattern
    cdef regex* time_pattern
    cdef regex* contains_digit

    def __cinit__( self, filename, alpha = 0.7, case_sensitive = False,
                   n_label_type = 0 ):
        self.word2idx = {}
        self.word2fofe = {}
        self.alpha = alpha
        self.case_sensitive = case_sensitive
        self.date_pattern_1 = new regex( r"^(?:(?:31(\/|-|\.)(?:0?[13578]|1[02]|(?:Jan|Mar|May|Jul|Aug|Oct|Dec)))\1|(?:(?:29|30)(\/|-|\.)(?:0?[1,3-9]|1[0-2]|(?:Jan|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.)(?:0?2|(?:Feb))\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.)(?:(?:0?[1-9]|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep))|(?:1[0-2]|(?:Oct|Nov|Dec)))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$" )
        self.date_pattern_2 = new regex( r"^(((\d{4}(\/|-|\.)((0[13578](\/|-|\.)|1[02](\/|-|\.))(0[1-9]|[12]\d|3[01])|(0[13456789](\/|-|\.)|1[012](\/|-|\.))(0[1-9]|[12]\d|30)|02(\/|-|\.)(0[1-9]|1\d|2[0-8])))|((([02468][048]|[13579][26])00|\d{2}([13579][26]|0[48]|[2468][048])))(\/|-|\.)02(\/|-|\.)29)){0,10}$" )
        self.number_pattern = new regex( r"^(\+|-)?(([1-9]\d{0,2}(,\d{3})*)|([1-9]\d*)|0)(\.\d+)?$" )
        self.phone_pattern = new regex( r"^(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?$" )
        self.time_pattern = new regex( r"^(?:(?:([01]?\d|2[0-3]):)?([0-5]?\d):)?([0-5]?\d)$" )
        self.contains_digit = new regex( r"^.*[0-9].*$" )

        with codecs.open( filename ) as word_file:
            for line in word_file:
                word = line.strip().split()[0]
                idx = len(self.word2idx)
                self.word2idx[word] = idx

        logger.info( '%d words' % len(self.word2idx) )

        # 2nd pass, make room for labels
        self.n_word = len(self.word2idx) - n_label_type - (2 if self.case_sensitive else 1) - 1
        if n_label_type > 0:
            for w in self.word2idx.keys():
                if w != '<unk>' and w != '<UNK>' \
                        and self.word2idx[w] >= self.n_word:
                    self.word2idx.pop(w, None)

        self.pad_idx = self.n_word


    def __len__( self ):
        return self.n_word


    cdef sentence2indices( self, sentence, vector[int]& numeric ):
        cdef string s
        cdef int i
        cdef int n = len( sentence )
        cdef int unk = self.word2idx['<unk>']
        cdef int UNK = self.word2idx.get( '<UNK>', unk )
        numeric.resize( n )
        for i, w in enumerate(sentence):
            s = w.lower()
            if regex_match( s, self.contains_digit[0] ):
                if regex_match( s, self.number_pattern[0] ):
                    numeric[i] = self.word2idx.get('<numeric-value>', unk)
                elif regex_match( s, self.date_pattern_1[0] ) or \
                                regex_match( w, self.date_pattern_2[0] ):
                    numeric[i] = self.word2idx.get('<date-value>', unk)
                elif regex_match( s, self.time_pattern[0] ):
                    numeric[i] = self.word2idx.get('<time-value>', unk)
                elif regex_match( s, self.phone_pattern[0] ):
                    numeric[i] = self.word2idx.get('<phone-value>', unk)
                else:
                    numeric[i] = self.word2idx.get('<contains-digit>', unk)
            else:
                if self.case_sensitive:
                    if w == w.lower():
                        numeric[i] = self.word2idx.get( w, unk )
                    else:
                        numeric[i] = self.word2idx.get( w, UNK )
                else:
                    numeric[i] = self.word2idx.get( s, unk )

    def __dealloc__( self ):
        del self.date_pattern_1
        del self.date_pattern_2
        del self.time_pattern
        del self.number_pattern
        del self.phone_pattern
        del self.contains_digit


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def char_fofe_of_word( self, word ):
        if len( self.word2fofe ) > 2 * len(self.word2idx):
            self.word2fofe = {}
        if word in self.word2fofe:
            return self.word2fofe[word]
        else:
            lfofe, coeff = numpy.zeros((128,), numpy.float32), 1
            for c in reversed(word):
                i = ord(c) if 0 < ord(c) < 128 else 0
                lfofe[i] += numpy.float32(coeff)
                coeff *= self.alpha
            rfofe, coeff = numpy.zeros((128,), numpy.float32), 1
            for c in word:
                i = ord(c) if 0 < ord(c) < 128 else 0
                rfofe[i] += numpy.float32(coeff)
                coeff *= self.alpha
            self.word2fofe[word] = [lfofe, rfofe]
            return [lfofe, rfofe]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def char_fofe_of_phrase( self, phrase ):
        cdef int i, n = len(phrase)
        fofe = [ self.char_fofe_of_word(w) for w in phrase ]
        lfofe = fofe[0][0].copy()
        rfofe = fofe[n - 1][1].copy()
        for i in range(1, n):
            lfofe *= self.alpha ** len(phrase[i - 1])
            lfofe += fofe[i][0]
        for i in reversed(range(n - 1)):
            rfofe *= self.alpha ** len(phrase[i + 1])
            rfofe += fofe[i][1]
        return lfofe, rfofe


    def padding_index( self ):
        return self.pad_idx


################################################################################


class chinese_word_vocab( vocabulary ):
    """
    The legancy code is strongly-typed. Polymorphism is the fastest fix. 
    """
    def __init__( self, filename, n_label_type = 0 ):
        self.number = re.compile( r"^(?=[^A-Za-z]+$).*[0-9].*$".encode('utf8') )

        with codecs.open( filename, 'rb', 'utf8' ) as fp:
            self.idx2word = [ w.strip() for w in fp.read().strip().split( u'\n' ) ]
        self.word2idx = { w:i for (i,w) in enumerate( self.idx2word ) }

        # 2nd pass, make room for labels
        self.n_word = len(self.word2idx) - n_label_type - 1 - 1
        if n_label_type > 0:
            for w in self.word2idx.keys():
                if w != '<unk>' and self.word2idx[w] >= self.n_word:
                    self.word2idx.pop(w, None)

        self.pad_idx = self.n_word



    def sentence2indices( self, sentence ):
        w_unk = self.word2idx[u'<unk>']
        result = [ self.word2idx.get(w, w_unk) for w in \
                   imap( lambda w: u'<numeric>' if re.match(self.number, w) else w, sentence ) ]
        return result


    def char_fofe_of_word( self ):
        raise AttributeError( "'chinese_word_vocab' does not provide 'char_fofe_of_word'" )


    def char_fofe_of_phrase( self ):
        raise AttributeError( "'chinese_word_vocab' does not provide 'char_fofe_of_phrase'" )


    def padding_index( self ):
        return self.pad_idx

################################################################################


cdef class processed_sentence:
    """
    Any object of this class should not be instantiated outside this module.
    """
    cdef public vector[int] numeric
    cdef readonly vector[string] sentence
    cdef readonly vector[vector[int]] left_context_idx
    cdef readonly vector[vector[float]] left_context_data
    cdef readonly vector[vector[int]] right_context_idx
    cdef readonly vector[vector[float]] right_context_data

    def __init__( self, sentence, numericizer, 
                  a = 0.7, language = 'eng', label1st = None ):
        """
        Parameters
        ----------
            sentence : list of str
            numericizer : vocabulary
            a : float
                word-level forgetting factor
            language : eng
                either 'eng', 'cmn' or 'spa'
            label1st : list
                labels from 1st pass
        """

        cdef vocabulary vocab

        if language != 'cmn':
            for w in sentence:
                self.sentence.push_back( u''.join( c if ord(c) < 128 else chr(ord(c) % 32) for c in list(w) ) )
            vocab = numericizer
            vocab.sentence2indices( self.sentence, self.numeric )
        else:
            self.numeric = numericizer.sentence2indices( sentence )

        cdef vector[int] idx_buffer
        cdef vector[float] data_buffer
        cdef ordered_map[int,float] left_context
        cdef ordered_map[int,float] right_context
        cdef ordered_map[int,float].iterator map_itr
        cdef float alpha = a
        cdef int i
        cdef int idx
        cdef int n = self.numeric.size()

        # used to record the represnetation up to the last
        cdef bint is2ndPass = (label1st is not None)
        cdef ordered_map[int,int] boe
        cdef ordered_map[int,int] eoe
        cdef ordered_map[int,float] left_context_2nd
        cdef ordered_map[int,float] right_context_2nd
        cdef int n_word = len(numericizer)

        if is2ndPass:
            # TODO, remove nested mention
            boe = dict(zip(label1st[0], label1st[2]))
            eoe = dict(zip(label1st[1], label1st[2]))

        with nogil: 
            for i in range( n ):
                if boe.find(i) != boe.end():
                    left_context_2nd = left_context

                if eoe.find(i + 1) == eoe.end():
                    idx = self.numeric[i]
                else:
                    idx = n_word + eoe[i + 1]
                    left_context = left_context_2nd

                if i == 0:
                    left_context[idx] = 1
                else:
                    map_itr = left_context.begin()
                    while map_itr != left_context.end():
                        left_context[dereference(map_itr).first] = dereference(map_itr).second * alpha
                        preincrement(map_itr)
                    if left_context.find(idx) != left_context.end():
                        left_context[idx] += 1
                    else:
                        left_context[idx] = 1

                idx_buffer.clear()
                data_buffer.clear()
                map_itr = left_context.begin()
                while map_itr != left_context.end():
                    idx_buffer.push_back( dereference(map_itr).first )
                    data_buffer.push_back( dereference(map_itr).second )
                    preincrement(map_itr)
                self.left_context_idx.push_back( idx_buffer )
                self.left_context_data.push_back( data_buffer )

            for i in reversed( range( n ) ):
                if eoe.find(i + 1) != eoe.end():
                    right_context_2nd = right_context

                if boe.find(i) == boe.end():
                    idx = self.numeric[i]
                else:
                    idx = n_word + boe[i]
                    right_context = right_context_2nd

                if i == n - 1:
                    right_context[idx] = 1
                else:
                    map_itr = right_context.begin()
                    while map_itr != right_context.end():
                        right_context[dereference(map_itr).first] = dereference(map_itr).second * alpha
                        preincrement(map_itr)
                    if right_context.find(idx) != right_context.end():
                        right_context[idx] += 1
                    else:
                        right_context[idx] = 1

                idx_buffer.clear()
                data_buffer.clear()
                map_itr = right_context.begin()
                while map_itr != right_context.end():
                    idx_buffer.push_back( dereference(map_itr).first )
                    data_buffer.push_back( dereference(map_itr).second )
                    preincrement(map_itr)
                self.right_context_idx.push_back( idx_buffer )
                self.right_context_data.push_back( data_buffer )

            reverse( self.right_context_idx.begin(), self.right_context_idx.end() )
            reverse( self.right_context_data.begin(), self.right_context_data.end() )



    cdef insert_left_fofe( self, int pos, int row_id, 
                           vector[int]& indices, vector[float]& values ):
        """ help to construct mini-batch """
        cdef int n = self.left_context_idx[pos].size()
        with nogil:
            for j in range( n ):
                values.push_back( self.left_context_data[pos][j] )
                indices.push_back( row_id )
                indices.push_back( self.left_context_idx[pos][j] )


    cdef insert_right_fofe( self, int pos, int row_id, 
                           vector[int]& indices, vector[float]& values ):
        """ help to construct mini-batch """
        cdef int i
        cdef int n = self.right_context_idx[pos].size()
        with nogil:
            for i in range( n ):
                values.push_back( self.right_context_data[pos][i] )
                indices.push_back( row_id )
                indices.push_back( self.right_context_idx[pos][i] )


    cdef insert_bow( self, int begin_idx, int end_idx,
                     int row_id, vector[int]& indices ):
        """ help to construct mini-batch """
        cdef int i
        with nogil:
            for i in range( begin_idx, end_idx ):
                indices.push_back( row_id )
                indices.push_back( self.numeric[i] )



################################################################################


cdef class example:
    cdef readonly int sentence_id
    cdef readonly int begin_idx
    cdef readonly int end_idx
    cdef readonly int label
    cdef readonly numpy.ndarray gazetteer

    def __init__( self, sentence_id, begin_idx, end_idx, label, gazetteer = None ):
        self.sentence_id = sentence_id
        self.begin_idx = begin_idx
        self.end_idx = end_idx
        self.label = label
        self.gazetteer = gazetteer
        
        
        
################################################################################


cdef void bigram_char_fofe( string phrase, 
                            vector[float]& values, vector[int]& indices,
                            float alpha, int row_id = 0 ) nogil:
    # printable characters range from 32 (inclusive) to 127 (exclusive), totalling 127 - 32 = 95
    # anything out of this range is considered OOV
    cdef int i, c
    cdef float coef = 1
    cdef vector[int] char_array
    cdef ordered_map[int,float] fofe
    cdef ordered_map[int,float].iterator map_itr

    for i in range( phrase.size() ):
        c = <int>phrase[i];
        if 32 <= c < 127:
            char_array.push_back( c - 32 )
        else:
            char_array.push_back( 95 )
        
    for i in reversed( range( char_array.size() - 1 ) ):
        c = char_array[i] * 96 + char_array[i + 1]
        if fofe.find( c ) != fofe.end():
            fofe[c] += coef
        else:
            fofe[c] = coef

    map_itr = fofe.begin()
    while map_itr != fofe.end():
        indices.push_back( row_id )
        indices.push_back( dereference(map_itr).first )
        values.push_back( dereference(map_itr).second )
        preincrement( map_itr )


################################################################################



class batch_constructor:
    def __init__( self, parser, 
                  numericizer1, numericizer2,
                  gazetteer = None, window = 7, alpha = 0.7, 
                  n_label_type = 4, language = 'eng',
                  is2ndPass = False ):
        """
        Parameters
        ----------
            parser : iterable
                Likes of KBP2015, CoNLL2003, where each element is a tuple of size 4

            numericizer1 : vocabulary
                case-insensitive vocabulary

            numericizer2 : vocabulary
                case-sensitive vocabulary

            gazetteer : list of set
                Likes of gazetteer, gazetteer[i] contains the known NER of the ith mention type

            window : int
                the maximum length of a mention
                
            n_label_type : int
                number of mention types excluding O. For example, CoNLL2003 has 4 mention types,
                namely, PER, ORG, LOC and MISC.

            language : str
                either 'eng', 'cmn' or 'spa'

            is2ndPass : bool
                enable 2nd pass if true
        """
        assert language in { 'eng', 'cmn', 'spa' }
        self.language = language

        # case-insensitive sentence set if language in { 'eng', 'spa' }
        # sequence at char level
        self.sentence1 = []

        # case-sensitive sentence set if language in { 'eng', 'spa' }
        # sequence at word level
        self.sentence2 = []

        self.example = []
        self.positive = []
        self.overlap = []
        self.disjoint = []

        self.is2ndPass = is2ndPass

        # luckily that 'batch_constructor' is not strongly-typed
        # it is OK that these two data members hold garbage value when parsing Chinese
        self.numericizer1 = numericizer1    # case-insensitive / char-level
        self.numericizer2 = numericizer2    # case-sensitive / word-level

        self.gazetteer = gazetteer
        self.n_label_type = n_label_type

        cdef int i, j, k
        cdef bint unsure

        for sentence, ner_begin, ner_end, ner_label in parser:
            ner_begin = numpy.asarray(ner_begin, dtype = numpy.int32)
            ner_end = numpy.asarray(ner_end, dtype = numpy.int32)
            ner_label = numpy.asarray(ner_label, dtype = numpy.int32)
            label1st_powerset = []
            # if self.is2ndPass and len(ner_label) > 0:
            #     powerItr = combinations(numpy.arange(len(ner_label)), len(ner_label) - 1)
            #     # powerItr = chain.from_iterable( 
            #     #             combinations(numpy.arange(len(ner_label)), x) \
            #     #                         for x in xrange(len(ner_label) + 1) )
            #     # powerItr = reduce(lambda result, x: \
            #     #         result + [subset + [x] for subset in result], range(len(ner_label)), [[]])
            #     for label1st in powerItr:
            #         label1st = numpy.asarray(label1st, dtype = numpy.int32)
            #         label1st_powerset.append( (ner_begin[label1st], 
            #                                 ner_end[label1st], ner_label[label1st]) )
            
            label1st_powerset.append( (ner_begin, ner_end, ner_label) )

            for label1st in label1st_powerset:
                for i in range( len(sentence) ):
                    for j in range( i + 1, len(sentence) + 1 ):
                        unsure, found = False, False
                        if j - i > window:
                            break
                        label = n_label_type
                        # look for exact match
                        for k in range(len(ner_label)):
                            if i == ner_begin[k] and j == ner_end[k]:
                                label = ner_label[k]
                                if label < n_label_type:
                                    self.positive.append( len(self.example) )
                                else:
                                    unsure = True
                                found = True
                                break
                        # look for overlap
                        if not found:
                            for k in range(len(ner_label)):
                                if i < ner_end[k] and ner_begin[k] < j:
                                    label = n_label_type + 1
                                    self.overlap.append( len(self.example) )
                                    break
                        if unsure:
                            continue
                        if label == n_label_type:
                            self.disjoint.append( len(self.example) )
                        if label == n_label_type + 1:
                            label = n_label_type

                        gazetteer_match = numpy.zeros( (n_label_type + 1,), dtype = numpy.float32 )
                        if self.gazetteer is not None:
                            if language != 'cmn':
                                name = u' '.join(sentence[i:j])
                            else:
                                name = u''.join( w[:w.find(u'|iNCML|')] for w in sentence[i:j] )
                            for k, g in enumerate(self.gazetteer):
                                if name in g:
                                    gazetteer_match[k] = 1

                        self.example.append( example( len(self.sentence1), 
                                                      i, j, label ,gazetteer_match) )
                
                if not self.is2ndPass:
                    label1st = None

                if language != 'cmn': 
                    self.sentence1.append( processed_sentence( sentence, numericizer1, 
                                                               alpha, language,
                                                               label1st ) )
                    self.sentence2.append( processed_sentence( sentence, numericizer2, 
                                                               alpha, language,
                                                               label1st ) )
                else:
                    char_sequence, word_sequence = [], []
                    for token in sentence:
                        c, w = token.split( u'|iNCML|' )
                        char_sequence.append( c )
                        word_sequence.append( w )
                    self.sentence1.append( processed_sentence( char_sequence, numericizer1,
                                                               alpha, language,
                                                               label1st ) )
                    self.sentence2.append( processed_sentence( word_sequence, numericizer2,
                                                               alpha, language,
                                                               label1st ) )

        self.positive = numpy.asarray( self.positive, dtype = numpy.int32 )
        self.overlap = numpy.asarray( self.overlap, dtype = numpy.int32 )
        self.disjoint = numpy.asarray( self.disjoint, dtype = numpy.int32 )


    def __str__( self ):
        """
        Returns
        -------
            Return a string description of this object.
        """
        return ('%d sentences, %d (positive), %d (overlap), %d (disjoint)' % 
                (len(self.sentence1), 
                    self.positive.shape[0], self.overlap.shape[0], self.disjoint.shape[0]) )


    @cython.boundscheck(False)
    def mini_batch( self, int n_batch_size, 
                    bint shuffle_needed = True, float overlap_rate = 0.36, 
                    float disjoint_rate = 0.08, int feature_choice = 255, 
                    bint replace = False, int n_copy = 1  ):
        """
        The generator yields mini batches of size 'n_batch_size'. Based on 
        'feature_choice', the following features may be selected:
        1) case-insensitive left fofe including focus word(s)
        2) case-insensitive right fofe incuding focus word(s)

        Parameters
        ----------
            n_batch_size : int
                mini-batch size, the last mini-batch might be smaller

            shuffle_needed : bool
                During training, 'shuffle_needed' should be 'True' so as to 
                introduce randomization. During testing, 'shuffle_needed' should 
                be 'False' for alignment.

            overlap_rate : float
                sample rate of negative examples that overlaps with positive examples

            disjoint_rate : float 
                sample rate of negative examples that is disjoint with postiive examples

            replace : bool
                when 'overlap_rate' or 'disjoint_rate' is less than 1, sampling 
                is done with/without replacement if True/False

            feature_choice : int
                Look at 'Returns' for detailed description. If 'feature_choice' & (1 << ith) is 'True',

            n_copy : int
                how many times to chain this iterator

        Returns
        -------
            l1_values : 
            r1_values : 


        """
        cdef vector[float] l1_values    # case-insensitive left context fofe with focus words(s)
        cdef vector[float] r1_values    
        cdef vector[int] l1_indices     # case-insensitive right context fofe with focus word(s)
        cdef vector[int] r1_indices
        cdef vector[float] l2_values    # case-insensitive left context fofe without focus words(s)
        cdef vector[float] r2_values    
        cdef vector[int] l2_indices     # case-insensitive right context fofe without focus word(s)
        cdef vector[int] r2_indices
        cdef vector[int] bow1           # case-insensitive bow

        cdef vector[float] l3_values    # case-sensitive left context fofe with focus words(s)
        cdef vector[float] r3_values    
        cdef vector[int] l3_indices     # case-sensitive right context fofe with focus word(s)
        cdef vector[int] r3_indices
        cdef vector[float] l4_values    # case-sensitive left context fofe without focus words(s)
        cdef vector[float] r4_values    
        cdef vector[int] l4_indices     # case-sensitive right context fofe without focus word(s)
        cdef vector[int] r4_indices     
        cdef vector[int] bow2           # case-sensitive bow

        cdef vector[vector[int]] conv_idx
        cdef vector[int] conv_buff
        cdef string phrase, reversed_phrase
        
        cdef vector[float] lbc_values           # left bigram-char fofe
        cdef vector[int] lbc_indices
        
        cdef vector[float] rbc_values           # right bigram-char fofe
        cdef vector[int] rbc_indices

        cdef example next_example
        cdef processed_sentence sentence
        cdef vector[int] label
        cdef int i, j, k, begin_idx, end_idx
        cdef int cnt = 0
        cdef int n
        cdef int phrase_max_length = 10
        cdef float bigram_alpha

        has_char_feature = feature_choice & (64 | 128 | 512 | 1024)
        assert not has_char_feature or self.language != 'cmn', \
                'Chinese is modeled at character level. '

        if n_copy > 1:
            shuffle_needed = True
            replace = True

        dense_buffer = numpy.zeros( (n_batch_size, 513 + self.n_label_type), dtype = numpy.float32 )

        if len( self.disjoint ) > 0: 
            disjoint = numpy.random.choice( self.disjoint,
                                            size = numpy.int32( len(self.disjoint) * disjoint_rate * n_copy ),
                                            replace = replace )
        else:
            disjoint = numpy.asarray([]).astype( numpy.int32 )

        if len( self.overlap ) > 0:
            overlap = numpy.random.choice( self.overlap,
                                           size = numpy.int32( len(self.overlap) * overlap_rate * n_copy ),
                                           replace = replace )
        else:
            overlap = numpy.asarray([]).astype( numpy.int32 )

        candidate = numpy.concatenate( [ self.positive ] * n_copy + [ disjoint, overlap ] )

        if shuffle_needed:
            numpy.random.shuffle( candidate )
        else:
            candidate.sort()
        n = len(candidate)

        for i in range( n ):
            next_example = self.example[ candidate[i] ]
            begin_idx = next_example.begin_idx
            end_idx = next_example.end_idx

            sentence = self.sentence1[next_example.sentence_id]

            if self.language != 'cmn':
                phrase = ' '.join( sentence.sentence[begin_idx:end_idx] )
                reversed_phrase = ' '.join( sentence.sentence[begin_idx:end_idx] )[::-1]
                bigram_alpha = self.numericizer1.alpha

            with nogil:
                # char-level convolution indices
                # 1 padding at the begining and the end respectively
                if feature_choice & 512 > 0:
                    if phrase.size() + 2 > phrase_max_length:
                        phrase_max_length = phrase.size() + 2
                    conv_buff.clear()
                    conv_buff.push_back( 0 )
                    for k in range( phrase.size() ):
                        conv_buff.push_back( <int>phrase[k] )
                    conv_idx.push_back( conv_buff )
                    
                # bigram char-fofe
                if feature_choice & 1024 > 0:
                    bigram_char_fofe( phrase, lbc_values, lbc_indices, bigram_alpha, cnt )
                    bigram_char_fofe( reversed_phrase, rbc_values, rbc_indices, 
                                      bigram_alpha, cnt )

            # character-level fofe of focus word(s)

            if feature_choice & 64 > 0:
                
                left_c, right_c = self.numericizer1 \
                                      .char_fofe_of_phrase( sentence.sentence[begin_idx:end_idx] )
                
                # left_c, _ = self.numericizer1.char_fofe_of_word(
                #             ''.join( [ w[0] for w in sentence.sentence[:end_idx] ] ) )
                # _, right_c = self.numericizer1.char_fofe_of_word(
                #             ''.join( [ w[0] for w in sentence.sentence[begin_idx:] ] ) )
                
                dense_buffer[cnt,:128] = left_c
                dense_buffer[cnt,128:256] = right_c

            # character-level fofe of initial of focus word(s)

            if feature_choice & 128 > 0:
                left_init, right_init = self.numericizer1.char_fofe_of_word(
                            ''.join( [ w[0] for w in sentence.sentence[begin_idx:end_idx] ] ) )
                dense_buffer[cnt,256:384] = left_init
                dense_buffer[cnt,384:512] = right_init

            # gazetteer match

            if feature_choice & 256 > 0:
                dense_buffer[cnt,512:] = next_example.gazetteer

            label.push_back( next_example.label )

            ########## case-insensitive context with focus ##########

            if feature_choice & 1 > 0:
                sentence.insert_left_fofe( end_idx - 1, cnt, l1_indices, l1_values )
                sentence.insert_right_fofe( begin_idx, cnt, r1_indices, r1_values )

            ########## case-insensitive context without focus ##########

            if feature_choice & 2 > 0:
                if begin_idx != 0:
                    sentence.insert_left_fofe( begin_idx - 1, cnt, l2_indices, l2_values )

                if end_idx != sentence.numeric.size():
                    sentence.insert_right_fofe( end_idx, cnt, r2_indices, r2_values )

            ########## case-insensitive bow ##########

            if feature_choice & 4 > 0:
                sentence.insert_bow( begin_idx, end_idx, cnt, bow1 )

            # switch to case-sensitive
            sentence = self.sentence2[next_example.sentence_id]

            ########## case-sensitive context with focus ##########

            if feature_choice & 8 > 0:
                sentence.insert_left_fofe( end_idx - 1, cnt, l3_indices, l3_values )
                sentence.insert_right_fofe( begin_idx, cnt, r3_indices, r3_values )


            ########## case-sensitive context without focus ##########

            if feature_choice & 16 > 0:
                if begin_idx != 0:
                    sentence.insert_left_fofe( begin_idx - 1, cnt, l4_indices, l4_values )

                if end_idx != sentence.numeric.size():
                    sentence.insert_right_fofe( end_idx, cnt, r4_indices, r4_values )

            if feature_choice & 32 > 0:
                sentence.insert_bow( begin_idx, end_idx, cnt, bow2 )

            cnt += 1
            if cnt % n_batch_size == 0 or (i + 1) == len(candidate):
                with nogil:
                    if feature_choice & 512 > 0:
                        for k in range( conv_idx.size() ):
                            while conv_idx[k].size() < phrase_max_length:
                                conv_idx[k].push_back( 0 )
                            if conv_idx[k].size() > 128:
                                conv_idx[k].resize( 128 )

                # print 'i am right before yield statement, cnt = %d' % cnt

                yield   numpy.asarray( l1_values, dtype = numpy.float32 ),\
                        numpy.asarray( r1_values, dtype = numpy.float32 ),\
                        numpy.reshape( l1_indices, [-1, 2] ),\
                        numpy.reshape( r1_indices, [-1, 2] ),\
                        numpy.asarray( l2_values, dtype = numpy.float32 ),\
                        numpy.asarray( r2_values, dtype = numpy.float32 ),\
                        numpy.reshape( l2_indices, [-1, 2] ),\
                        numpy.reshape( r2_indices, [-1, 2] ),\
                        numpy.reshape( bow1, [-1, 2] ),\
                        numpy.asarray( l3_values, dtype = numpy.float32 ),\
                        numpy.asarray( r3_values, dtype = numpy.float32 ),\
                        numpy.reshape( l3_indices, [-1, 2] ),\
                        numpy.reshape( r3_indices, [-1, 2] ),\
                        numpy.asarray( l4_values, dtype = numpy.float32 ),\
                        numpy.asarray( r4_values, dtype = numpy.float32 ),\
                        numpy.reshape( l4_indices, [-1, 2] ),\
                        numpy.reshape( r4_indices, [-1, 2] ),\
                        numpy.reshape( bow2, [-1, 2] ),\
                        dense_buffer[:cnt].copy(),\
                        numpy.asarray( conv_idx ) if conv_idx.size() > 0 else numpy.empty((0,0), numpy.int64),\
                        numpy.asarray( lbc_values, dtype = numpy.float32 ), \
                        numpy.reshape( lbc_indices, [-1, 2]  ), \
                        numpy.asarray( rbc_values, dtype = numpy.float32 ), \
                        numpy.reshape( rbc_indices, [-1, 2]  ), \
                        numpy.asarray( label )

                with nogil:
                    cnt = 0
                    phrase_max_length = 10

                    l1_values.clear()
                    r1_values.clear()
                    l1_indices.clear()
                    r1_indices.clear()
                    l2_values.clear()
                    r2_values.clear()
                    l2_indices.clear()
                    r2_indices.clear()
                    bow1.clear()

                    l3_values.clear()
                    r3_values.clear()
                    l3_indices.clear()
                    r3_indices.clear()
                    l4_values.clear()
                    r4_values.clear()
                    l4_indices.clear()
                    r4_indices.clear()
                    bow2.clear()

                    conv_idx.clear()
                    label.clear()

                    lbc_indices.clear()
                    lbc_values.clear()
                    rbc_indices.clear()
                    rbc_values.clear()

                dense_buffer = numpy.zeros( (n_batch_size, 513 + self.n_label_type), 
                                            dtype = numpy.float32 )


    def mini_batch_multi_thread( self, int n_batch_size, 
                                 bint shuffle_needed = True, float overlap_rate = 0.36, 
                                 float disjoint_rate = 0.08, int feature_choice = 255, 
                                 bint replace = False, float timeout = -1, int n_copy = 1  ):
        """
        Same as self.mini_batch except that data preparation is done on the background
        """
        batch_generator = self.mini_batch( n_batch_size, shuffle_needed, 
                                           overlap_rate, disjoint_rate,
                                           feature_choice, replace )
        batch_buffer = Queue( maxsize = 256 )
        t = Thread( target = prepare_mini_batch, 
                    args = ( batch_generator, batch_buffer, timeout if timeout > 0 else None ) )
        t.daemon = True
        t.start()
        while True:
            next_batch = batch_buffer.get( True, timeout if timeout > 0 else None )
            if next_batch is not None:
                yield next_batch
            else:
                break


    def infinite_mini_batch_multi_thread( self, int n_batch_size, 
                                          bint shuffle_needed = True, float overlap_rate = 0.36, 
                                          float disjoint_rate = 0.08, int feature_choice = 255, 
                                          bint replace = True, float timeout = -1, int n_copy = 10  ):
        """
        Same as self.mini_batch_multi_thread except that sampling is done infinitely.
        """
        while True:
            for next_batch in self.mini_batch_multi_thread( n_batch_size, shuffle_needed, 
                                                            overlap_rate, disjoint_rate,
                                                            feature_choice, replace, timeout, n_copy ):
                if next_batch[-1].shape[0] == n_batch_size:
                    yield next_batch






################################################################################


def SampleGenerator( filename ):
    ner2idx = { 'PER' : 0, 'LOC' : 1, 'ORG' : 2, 'MISC' : 3, 'O' : 4 }
    sentence, beginOfNer, endOfNer, nerCls = [], [], [], []
    lastNer = 4

    corpus = open( filename, 'rb' )

    for line in corpus:
        line = line.strip()
        
        tokens = line.split()

        if len(tokens) > 1:
            word, ner = tokens[0], ner2idx[ tokens[-1].split('-')[-1] ]
            if ner != lastNer:
                if lastNer != 4:
                    endOfNer.append( len(sentence) )
                if ner != 4:
                    beginOfNer.append( len(sentence) )
                    nerCls.append( ner )
            lastNer = ner
            sentence.append( word )
        else:
            if len(sentence) > 0:
                if len(endOfNer) < len(beginOfNer):
                    endOfNer.append( len(sentence) )
                assert len(beginOfNer) == len(endOfNer)

                yield sentence, beginOfNer, endOfNer, nerCls

                sentence, beginOfNer, endOfNer, nerCls = [], [], [], []
                lastNer = 4

    corpus.close()



# def PredictionParser( dataset, result, ner_max_length, 
#                       reinterpret_threshold = 0, n_label_type = 4 ):
def PredictionParser( sample_generator, result, ner_max_length, 
                      reinterpret_threshold = 0, n_label_type = 4 ):
    """
    This function is modified from some legancy code. 'table' was designed for 
    visualization. 

    Parameters
    ----------
        sample_generator : iterable
            Likes of CoNLL2003 and KBP2015

        result : str
            path to a filename where each line is predicted class (in integer) 
            followed by the probabilities of each class

        ner_max_length: int
            maximum length of mention

        reinterpret_threshold: float
            NOT USED ANYMORE

        n_label_type : int
            numer of memtion types

    Yields
    ------
        s : list of str
            words in a sentence

        table : numpy.ndarray
            table[i][j - 1] is a pair of string represnetation of predicted class
            and the corresponding probability of s[i][j]

        estimate : tuple
            (begin,end,class) triples

        actual : tuple
            (begin,end,class) triples
    """
    if n_label_type == 4:
        idx2ner = [ 'PER', 'LOC', 'ORG', 'MISC', 'O' ]
    else:
        # idx2ner = [ 'PER_NAM', 'PER_NOM', 'ORG_NAM', 'GPE_NAM', 'LOC_NAM', 'FAC_NAM', 'TTL_NAM', 'O'  ]
        idx2ner = [ 'PER_NAM', 'ORG_NAM', 'GPE_NAM', 'LOC_NAM', 'FAC_NAM',
                    'PER_NOM', 'ORG_NOM', 'GPE_NOM', 'LOC_NOM', 'FAC_NOM',
                    'O' ]  

    # sg = SampleGenerator( dataset )
    if isinstance(result, str):
        fp = open( result, 'rb' )
    else:
        fp = result
    sg = sample_generator

    # @xmb 20160717
    lines, cnt = fp.readlines(), 0

    while True:
        s, boe, eoe, cls = sg.next()
        actual = set( zip(boe, eoe, cls) )

        table = numpy.empty((len(s), len(s)), dtype = object)
        table[:,:] = None #''
        estimate = set()
        actual = set( zip(boe, eoe, cls) )

        for i in xrange(len(s)):
            for j in xrange(i + 1, len(s) + 1):
                if j - i <= ner_max_length:
                    # @xmb 20160717
                    # line = fp.readline()
                    line = lines[cnt]
                    cnt += 1

                    tokens = line.strip().split()
                    predicted_label = int(tokens[1])
                    all_prob = numpy.asarray([ numpy.float32(x) for x in tokens[2:] ])

                    if predicted_label == n_label_type:
                        if all_prob[n_label_type] < reinterpret_threshold:
                            all_prob[n_label_type] = 0
                            all_prob /= all_prob.sum()
                            predicted_label = all_prob.argmax()

                    if predicted_label != n_label_type:
                        predicted, probability = idx2ner[predicted_label], all_prob[predicted_label]
                        table[i][j - 1] = (predicted, probability)
                        estimate.add( (i, j, predicted_label ) )

        yield s, table, estimate, actual

    if isinstance(result, str):
        fp.close()



def SentenceIterator( filename ):
    with open( filename, 'rb' ) as corpus:
        sentence = []
        for line in corpus:
            line = line.strip()
            if len(line) > 0:
                sentence.append( line )
            else:
                yield sentence
                sentence = []
    # with open( filename, 'rb' ) as corpus:
    #     sentences = corpus.read().strip().split( '\n\n' )
    #     for sent in sentences:
    #         yield sent.strip().split()


################################################################################

class CustomizedThreshold( object ):
    def update_state( self, previous_solution ):
        self.solution.append( previous_solution )

    def restore_state( self ):
        self.solution.pop()

    def keep( self, candidate, estimate, table, global_threshold ):
        b, e, c = candidate
        return table[b][e - 1][1] >= global_threshold



class ORGcoverGPE( CustomizedThreshold ):
    """
    Some ORGs are named by GPEs, e,g, University of Toronto.
    No improvement is shown in KBP2015
    """
    def __init__( self, gpe_covered_by_org ):
        self.gpe_covered_by_org = gpe_covered_by_org
        self.solution = [ set() ]

    def keep( self, candidate, estimate, table, global_threshold ):
        b, e, c = candidate
        if c == 2:
            for bb, ee, cc in self.solution[-1]:
                if bb <= b < e <= ee and cc == 1:
                    return table[b][e - 1][1] >= self.gpe_covered_by_org
        return table[b][e - 1][1] >= global_threshold



class IndividualThreshold( CustomizedThreshold ):
    """
    Numbers of labels are imbalanced. Assign an individual threshold to each class.
    F1 score increase by 0.5 to 0.6 in KBP2015 
    """
    def __init__( self, outer, inner = None ):
        self.outer = outer
        self.inner = inner
        self.solution = [ set() ]
    
    def keep( self, candidate, estimate, table, global_threshold ):
        b, e, c = candidate
        if len(self.solution[-1]) == 0:
            return table[b][e - 1][1] >= self.outer[c]
        else:
            return table[b][e - 1][1] >= self.inner if not isinstance( self.inner, list ) \
                    else table[b][e - 1][1] >= self.inner[c]

    def __str__( self ):
        return 'outer: %s' % str(self.outer)


################################################################################


def __merge_adjacient( estimate ):
    best, i = set(), 0
    while i < len(estimate):
        j = i + 1
        while j < len(estimate):
            if estimate[j][0] == estimate[j - 1][1] and \
               estimate[j][2] == estimate[j - 1][2]:
                j += 1
            else:
                break
        assert estimate[i][2] == estimate[j - 1][2]
        best.add( (estimate[i][0], estimate[j - 1][1], estimate[i][2]) )
        i = j
    estimate = best
    return estimate



def __decode_algo_1( sentence, estimate, table, threshold, callback = None ):
    """
    Highest scrore first 
    """
    if callback is None:
        removed = set( [ (b, e, c) for (b, e, c) in estimate if table[b][e - 1][1] < threshold ] )
    else:
        removed = set( [ (b, e, c) for (b, e, c) in estimate if \
                         not callback.keep( (b,e,c), estimate, table, threshold ) ] )

    for i in xrange(len(sentence)):
        candidate = [ (b, e, c) for (b, e, c) in estimate if b <= i < e ]
        candidate.sort( key = lambda x : table[x[0]][x[1] - 1][1] )
        if len(candidate) > 0: 
            candidate.pop()
        removed = removed | set( candidate )

    estimate = list(estimate - removed)
    estimate.sort( key = lambda x : x[0] )
    estimate = __merge_adjacient( estimate )
    return estimate



def __decode_algo_2( sentence, estimate, table, threshold, callback = None ):
    """
    longest coverage first
    """
    candidate, best = {}, []

    for (b, e, c) in estimate:
        if callback is None:
            if table[b][e - 1][1] >= threshold:
                candidate[(b, e)] = c
        else:
            if callback.keep( (b,e,c), estimate, table, threshold ):
                candidate[(b, e)] = c

    for i in xrange(len(sentence)):
        if (0, i + 1) in candidate:
            best.append( (1, [(0, i + 1)]) )
        else:
            best.append( (0, []) )
        for j in xrange(i):
            if (j + 1, i + 1) in candidate and best[j][0] + i - j > best[-1][0]:
                best[-1] = (best[j][0] + i - j, best[j][1] + [(j + 1, i + 1)])
        if i > 0 and best[-2][0] > best[-1][0]:
            best[-1] = best[-2]
    estimate = [ (b, e, candidate[(b, e)]) for (b, e) in best[-1][1] ]
    estimate = __merge_adjacient( estimate )
    return estimate



def __decode_algo_3( sentence, estimate, table, threshold, callback = None ):
    if callback is None:
        removed = set( [ (b, e, c) for (b, e, c) in estimate if table[b][e - 1][1] < threshold ] )
    else:
        removed = set( [ (b, e, c) for (b, e, c) in estimate if \
                         not callback.keep( (b,e,c), estimate, table, threshold ) ] )

    estimate = estimate - removed

    for (b1, e1, c1) in estimate:
        for (b2, e2, c2) in estimate:
            if b2 <= b1 < e1 < e2 or b2 < b1 < e1 <= e2:
                removed.add( (b1, e1, c1) )
    estimate = estimate - removed

    return __decode_algo_1( sentence, estimate, table, threshold, callback )



def decode( sentence, estimate, table, threshold, algorithm, callback = None ):
    """
    Parameters
    ----------
        sentence : list
            list of words

        estimate : iterable
            a groups of (begin,end,class) triples. 'begin' and 'end' is and
            inclusive-exclusive pair

        table : numpy.ndarray


        threshold : float
            0 <= 'threshold' <= 1, probability under which is cut

        algorithm : int or list
            1 highest first; 2 longest coverage; 3 subsumssion removal

    Returns
    -------
        estimate : list
            (begin,end,class) trples with less overlapping based on 'algorithm'
    """
    if not isinstance( algorithm, list ):
        if not isinstance( estimate, set ):
            estimate = set( estimate )
        assert algorithm in [1, 2, 3], 'only 3 algorithms are supported'

        if algorithm == 1:
            return __decode_algo_1( sentence, estimate, table, threshold, callback )
        if algorithm == 2:
            return __decode_algo_2( sentence, estimate, table, threshold, callback )
        if algorithm == 3:
            return __decode_algo_3( sentence, estimate, table, threshold, callback )

    else:
        assert isinstance( threshold, list ) and len(threshold) == len(algorithm),\
                '#threshold and #algorithm do not match'
        result = decode( sentence, estimate, table, threshold[0], algorithm[0], callback )

        if len( algorithm ) > 1:
            for b1,e1,c1 in copy.deepcopy( result ):
                candidate = set()
                for b2,e2,c2 in estimate:
                    if c1 != c2 and (b1 <= b2 < e2 < e1 or b1 < b2 < e2 <= e1):
                        candidate.add( (b2,e2,c2) )

                if callback is not None:
                    callback.update_state( result )

                # result |= decode( sentence, candidate, table, threshold[1:], algorithm[1:], callback )
                # some adjacient tokens may merge and the merged one may produce conflit
                candidate = decode( sentence, candidate, table, threshold[1:], algorithm[1:], callback )
                for b2,e2,c2 in copy.deepcopy( candidate ):
                    if b2 == b1 and e2 == e1:
                        candidate.remove( (b2,e2,c2) )
                result |= candidate

                if callback is not None:
                    callback.restore_state()
        return result



def evaluation( prediction_parser, threshold, algorithm, 
                conll2003out = None, analysis = None, sentence_iterator = None,
                n_label_type = 4, decoder_callback = None ):
    # analysis = open( trainer_output.split('.')[0] + '.error', 'wb' )

    si = sentence_iterator
    pp = prediction_parser
    info = ''

    if n_label_type == 4:
        idx2ner = [ 'PER', 'LOC', 'ORG', 'MISC', 'O' ]
    else:
        # idx2ner = [ 'PER_NAM', 'PER_NOM', 'ORG_NAM', 'GPE_NAM', 'LOC_NAM', 'FAC_NAM', 'TTL_NAM', 'O'  ]
        idx2ner = [ 'PER_NAM', 'ORG_NAM', 'GPE_NAM', 'LOC_NAM', 'FAC_NAM',
                    'PER_NOM', 'ORG_NOM', 'GPE_NOM', 'LOC_NOM', 'FAC_NOM',
                    'O' ]  

    # each type maintains its own 'true-positive', 'false-positive' and 'false-negative' counts 
    true_positive, false_positive, false_negative = \
        [ 0 ] * n_label_type, [ 0 ] * n_label_type, [ 0 ] * n_label_type

    for sentence, table, estimate, actual in pp:
        # 'sorted_est' also serves as a copy of 'estimate' before anything is applied
        sorted_est = [ (b, e, idx2ner[c], table[b][e - 1][1]) for (b, e, c) in estimate ]

        estimate = decode( sentence, estimate, table, 
                           threshold, algorithm, decoder_callback )

        if analysis is not None and set(estimate) != set(actual):
            # print >> analysis, zip( range(len(sentence)), sentence )
            sorted_est.sort( key = lambda x : x[3], reverse = True )
            print >> analysis, '  '.join( [ w for w in sentence ] )
            print >> analysis, ''.join( [ ('%%-%dd' % (len(w) + 2)) % l for (l, w) in enumerate(sentence) ] )
            print >> analysis, '%10s' % 'raw-out: ', sorted_est
            print >> analysis, '%10s' % 'estimate: ', [ (b, e, idx2ner[c]) for (b, e, c) in estimate ]
            print >> analysis, '%10s' % 'actual: ', [ (b, e, idx2ner[c]) for (b, e, c) in actual ]

        estimate = set(estimate)

        for x in xrange( len(true_positive) ):
            # true_positive += len( estimate & actual )
            # false_positive += len( estimate - actual )
            # false_negative += len( actual - estimate )
            true_positive[x] += len( [ (b,e,c) for (b,e,c) in estimate & actual if c == x ] )
            false_positive[x] += len( [ (b,e,c) for (b,e,c) in estimate - actual if c == x ] )
            false_negative[x] += len( [ (b,e,c) for (b,e,c) in actual - estimate if c == x ] )

        if analysis is not None and set(estimate) != set(actual):
            print >> analysis, 'false-positive: ', [ (b, e, idx2ner[c]) for (b, e, c) in (estimate - actual)]
            print >> analysis, 'false-negative: ', [ (b, e, idx2ner[c]) for (b, e, c) in (actual - estimate)]
            print >> analysis

        # for CoNLL2003 output
        if si is not None:
            original = si.next()
            assert( len(original) == len(sentence) )
            tag = [ 'O' ] * len(sentence)
            for b, e, c in estimate:
                x = 'I-' + idx2ner[c]
                for i in xrange(b, e):
                    tag[i] = x
            for o, t in zip( original, tag ):
                if conll2003out:
                    print >> conll2003out, o, t
            if conll2003out:
                print >> conll2003out

    for x in xrange( len(true_positive) ):
        if true_positive[x] != 0:
            precision = float(true_positive[x]) / float(true_positive[x] + false_positive[x])
            recall = float(true_positive[x]) / float(true_positive[x] + false_negative[x])
            f_beta = 2.0 * precision * recall / (precision + recall)
        else:
            precision, recall, f_beta = 0.0, 0.0, 0.0
        info += '%12s  precision: %.2f%%, recall: %.2f%%, FB1: %.2f\n' % \
                            (idx2ner[x], precision * 100, recall * 100, f_beta * 100)

    true_positive, false_positive, false_negative = \
            sum(true_positive), sum(false_positive), sum(false_negative)
    if true_positive != 0:
        precision = float(true_positive) / float(true_positive + false_positive)
        recall = float(true_positive) / float(true_positive + false_negative)
        f_beta = 2.0 * precision * recall / (precision + recall)
    else:
        precision, recall, f_beta = 0.0, 0.0, 0.0
    info = '%-12s  precision: %.2f%%, recall: %.2f%%, FB1: %.2f\n' % \
                        ('OVERALL', precision * 100, recall * 100, f_beta * 100) + info

    if analysis is not None:
        # print >> analysis, 'precision: %f, recall: %f, F-beta: %f' % ( precision, recall, f_beta )
        print >> analysis, info
        analysis.close()

    return precision, recall, f_beta, info



################################################################################


def distant_supervision_parser( sentence_file, tag_file, 
                                start = 0, stop = None, step = 1,
                                mode = 'KBP',
                                escape_brackets = True ):
    if mode == 'KBP':
        str2idx = {
            '<PER>' : 0, '<ORG>' : 1, '<GPE>' : 2, 
            '<LOC>' : 3, '<FAC>' : 4, '<TTL>' : 5,
            '<MISC>' : 12, '<UNSURE>' : 13
        }
    else:
        # wiki data is mapped with KBP standard. 
        # this is a quick attempt to keep the overlapping part based on CoNLL2003
        str2idx = { 
            '<PER>' : 0, '<LOC>' : 1, '<ORG>' : 2, 
            '<GPE>' : 8, '<FAC>' : 8, '<TTL>' : 8,
            '<MISC>' : 8, '<UNSURE>' : 9
        }

    escape = {
        '(' : '-LRB-',
        ')' : '-RRB-',
        '[' : '-LSB-',
        ']' : '-RSB-',
        '{' : '-LCB-',
        '}' : '-RCB-'
    }


    with codecs.open( sentence_file, 'rb', 'utf8' ) as sentences, \
         codecs.open( tag_file, 'rb', 'utf8' ) as tags:

        for sentence, tag in islice( izip( sentences, tags ),
                                     start, stop, step ):
            if sentence.startswith( u'<p>' ) or \
                    sentence.startswith( u'</p>' ) or \
                    sentence.startswith( u'**********' ):
                continue

            # parse a sentence
            boe, eoe, loe, to_keep = [], [], [], False
            for x in tag.split():
                tokens = x.split( u',' )
                boe.append( int(tokens[0]) )
                eoe.append( int(tokens[1]) )
                loe.append( str2idx[tokens[2]] )
                if loe[-1] < 10:
                    to_keep = True

            if to_keep:
                sent = sentence.split()
                if escape_brackets:
                    sent = [ escape.get(w, w) for w in sent ]
                for b, e in zip( boe, eoe ):
                    # assert 0 <= b < e <= len(sent)
                    if not 0 <= b < e <= len(sent):
                        logger.exception( sentence )
                        logger.exception( tag )
                        to_keep = False
                        break

            if to_keep:
                for i,w in enumerate( sent ):
                    sent[i] = u''.join( c if 0 <= ord(c) < 128 \
                                          else chr(0) for c in list(w) )
                yield sent, boe, eoe, loe


################################################################################


cdef class processed_sentence_v2:
    cdef readonly vector[int] numeric
    cdef readonly vector[string] sentence
    cdef readonly vector[vector[int]] left2nd
    cdef readonly vector[vector[int]] right2nd
    cdef readonly bint is_2nd_pass


    def __init__( self, sentence, numericizer, 
                  language = 'eng', label1st = None ):
        cdef vocabulary vocab
        if language != 'cmn':
            for w in sentence:
                self.sentence.push_back(
                    u''.join( c if ord(c) < 128 else chr(ord(c) % 32) for c in list(w) )
                )
            vocab = numericizer
            vocab.sentence2indices( self.sentence, self.numeric )
        else:
            self.numeric = numericizer.sentence2indices( sentence )

        self.is_2nd_pass = (label1st is not None)

        cdef ordered_map[int,int] boe
        cdef ordered_map[int,int] eoe
        cdef vector[int] left_context
        cdef vector[int] right_context
        cdef vector[int] left_buff
        cdef vector[int] right_buff
        cdef int i
        cdef int idx
        cdef int n = self.numeric.size()
        cdef int n_word = len(numericizer)

        if self.is_2nd_pass:
            boe = dict(zip(label1st[0], label1st[2]))
            eoe = dict(zip(label1st[1], label1st[2]))

            with nogil:
                for i in range( n ):
                    if boe.find(i) != boe.end():
                        left_buff = left_context

                    if eoe.find(i + 1) == eoe.end():
                        idx = self.numeric[i]
                    else:
                        idx = n_word + eoe[i + 1]
                        left_context = left_buff

                    left_context.push_back( idx )
                    self.left2nd.push_back( left_context )

                for i in reversed( range( n ) ):
                    if eoe.find(i + 1) != eoe.end():
                        right_buff = right_context

                    if boe.find(i) == boe.end():
                        idx = self.numeric[i]
                    else:
                        idx = n_word + boe[i]
                        right_context = right_buff

                    right_context.push_back( idx )
                    self.right2nd.push_back( right_context ) 

                reverse( self.right2nd.begin(), self.right2nd.end() )


    @cython.boundscheck(False)
    cdef int insert_left( self, int pos, int[:] context ) nogil:
        cdef int i
        cdef int length = context.shape[0]
        if self.is_2nd_pass:
            if self.left2nd[pos].size() < length:
                length = self.left2nd[pos].size()
            for i in range(length):
                context[i] = self.left2nd[pos][i]
        else:
            if pos + 1 < length:
                length = pos + 1
            for i in range(length):
                context[i] = self.numeric[i]
        return length

        
    @cython.boundscheck(False)
    cdef int insert_right( self, int pos, int[:] context ) nogil:
        cdef int i
        cdef int length = context.shape[0]
        if self.is_2nd_pass:
            if self.right2nd[pos].size() < length:
                length = self.right2nd[pos].size()
            for i in range(length):
                context[i] = self.right2nd[pos][i]
        else:
            if self.numeric.size() - pos < length:
                length = self.numeric.size() - pos
            for i in range(length):
                context[i] = self.numeric[i + pos]
        return length


    @cython.boundscheck(False)
    cdef int insert_bow( self, int begin_idx, int end_idx, int[:] bow ) nogil:
        cdef int i
        cdef int length = end_idx - begin_idx
        if self.is_2nd_pass:
            return 0
        else:
            for i in range(length):
                bow[i] = self.numeric[i + begin_idx]
            return length
                            




class batch_constructor_v2:
    def __init__( self, parser, 
                  numericizer1, numericizer2,
                  gazetteer = None, window = 7, 
                  n_label_type = 4, language = 'eng',
                  is_2nd_pass = False ):
        assert language in { 'eng', 'cmn', 'spa' }
        self.language = language

        # case-insensitive sentence set if language in { 'eng', 'spa' }
        # sequence at char level
        self.sentence1 = []

        # case-sensitive sentence set if language in { 'eng', 'spa' }
        # sequence at word level
        self.sentence2 = []

        self.example = []
        self.positive = []
        self.overlap = []
        self.disjoint = []

        self.is2ndPass = is_2nd_pass

        # luckily that 'batch_constructor' is not strongly-typed
        # it is OK that these two data members hold garbage value when parsing Chinese
        self.numericizer1 = numericizer1    # case-insensitive / char-level
        self.numericizer2 = numericizer2    # case-sensitive / word-level

        self.gazetteer = gazetteer
        self.n_label_type = n_label_type

        cdef int i, j, k
        cdef bint unsure

        for sentence, ner_begin, ner_end, ner_label in parser:
            ner_begin = numpy.asarray(ner_begin, dtype = numpy.int32)
            ner_end = numpy.asarray(ner_end, dtype = numpy.int32)
            ner_label = numpy.asarray(ner_label, dtype = numpy.int32)
            label1st_powerset = []
            
            label1st_powerset.append( (ner_begin, ner_end, ner_label) )

            for label1st in label1st_powerset:
                for i in range( len(sentence) ):
                    for j in range( i + 1, len(sentence) + 1 ):
                        unsure, found = False, False
                        if j - i > window:
                            break
                        label = n_label_type
                        # look for exact match
                        for k in range(len(ner_label)):
                            if i == ner_begin[k] and j == ner_end[k]:
                                label = ner_label[k]
                                if label < n_label_type:
                                    self.positive.append( len(self.example) )
                                else:
                                    unsure = True
                                found = True
                                break
                        # look for overlap
                        if not found:
                            for k in range(len(ner_label)):
                                if i < ner_end[k] and ner_begin[k] < j:
                                    label = n_label_type + 1
                                    self.overlap.append( len(self.example) )
                                    break
                        if unsure:
                            continue
                        if label == n_label_type:
                            self.disjoint.append( len(self.example) )
                        if label == n_label_type + 1:
                            label = n_label_type

                        gazetteer_match = []
                        if self.gazetteer is not None:
                            if language != 'cmn':
                                name = u' '.join(sentence[i:j])
                            else:
                                name = u''.join( w[:w.find(u'|iNCML|')] for w in sentence[i:j] )
                            for k, g in enumerate(self.gazetteer):
                                if name in g:
                                    gazetteer_match.append(k)

                        self.example.append( 
                            example( 
                                len(self.sentence1), i, j, label ,
                                numpy.asarray(
                                    gazetteer_match,
                                    dtype = numpy.int32
                                )
                            ) 
                        )
                
                if not self.is2ndPass:
                    label1st = None

                if language != 'cmn': 
                    self.sentence1.append( 
                        processed_sentence_v2( 
                            sentence, 
                            numericizer1, 
                            language = language,
                            label1st = label1st
                        )
                    )
                    self.sentence2.append( 
                        processed_sentence_v2( 
                            sentence, 
                            numericizer2, 
                            language = language,
                            label1st = label1st
                        ) 
                    )
                else:
                    char_sequence, word_sequence = [], []
                    for token in sentence:
                        c, w = token.split( u'|iNCML|' )
                        char_sequence.append( c )
                        word_sequence.append( w )
                    self.sentence1.append( 
                        processed_sentence_v2( 
                            char_sequence, 
                            numericizer1,
                            language = language,
                            label1st = label1st 
                        ) 
                    )
                    self.sentence2.append( 
                        processed_sentence_v2( 
                            word_sequence, 
                            numericizer2,
                            language = language,
                            label1st = label1st
                        ) 
                    )

        self.positive = numpy.asarray( self.positive, dtype = numpy.int32 )
        self.overlap = numpy.asarray( self.overlap, dtype = numpy.int32 )
        self.disjoint = numpy.asarray( self.disjoint, dtype = numpy.int32 )


    @cython.boundscheck(False)
    def mini_batch( self, int n_batch_size, 
                    bint shuffle_needed = True, float overlap_rate = 0.36, 
                    float disjoint_rate = 0.08, int feature_choice = 255, 
                    bint replace = False, int n_copy = 1,
                    int context_limit = 64 ):

        lw1, rw1, lw2, rw2, lw3, rw3, lw4, rw4, \
        bow1, bow2 = [
            numpy.ones(
                (n_batch_size, context_limit),
                numpy.int32
            ) * (-1) for _ in xrange(10)
        ]
        lc, rc = [
            numpy.ones( 
                (n_batch_size, context_limit * 2),
                numpy.int32
            ) * 127 for _ in xrange(2)
        ]

        cdef int[:,:] lw1v = lw1
        cdef int[:,:] rw1v = rw1
        cdef int[:,:] lw2v = lw2
        cdef int[:,:] rw2v = rw2
        cdef int[:,:] lw3v = lw3
        cdef int[:,:] rw3v = rw3
        cdef int[:,:] lw4v = lw4
        cdef int[:,:] rw4v = rw4
        cdef int[:,:] bow1v = bow1
        cdef int[:,:] bow2v = bow2
        cdef int[:,:] lcv = lc
        cdef int[:,:] rcv = rc
        lcv[:,:] = 127
        rcv[:,:] = 127

        cdef int lw1len = 1
        cdef int rw1len = 1
        cdef int lw2len = 1
        cdef int rw2len = 1
        cdef int lw3len = 1
        cdef int rw3len = 1
        cdef int lw4len = 1
        cdef int rw4len = 1
        cdef int bowlen = 1
        cdef int clen = 10

        cdef example next_example
        cdef processed_sentence_v2 sentence
        cdef vector[int] label
        cdef int i, j, k, begin_idx, end_idx
        cdef int cnt = 0
        cdef int n
        cdef string phrase
        cdef int phrase_cpy_len
        cdef int [:] phrase_view


        gaz_buff = numpy.zeros(
            (n_batch_size, 1 + self.n_label_type),
            dtype = numpy.float32
        )
        cdef float[:,:] gaz_view = gaz_buff

        has_char_feature = feature_choice & (64 | 128 | 512 | 1024)
        assert not has_char_feature or self.language != 'cmn', \
                'Chinese is modeled at character level. '

        if n_copy > 1:
            shuffle_needed = True
            replace = True

        if len( self.disjoint ) > 0: 
            disjoint = numpy.random.choice( 
                self.disjoint,
                size = numpy.int32( len(self.disjoint) * disjoint_rate * n_copy ),
                replace = replace 
            )
        else:
            disjoint = numpy.asarray([]).astype( numpy.int32 )

        if len( self.overlap ) > 0:
            overlap = numpy.random.choice( 
                self.overlap,
                size = numpy.int32( len(self.overlap) * overlap_rate * n_copy ),
                replace = replace 
            )
        else:
            overlap = numpy.asarray([]).astype( numpy.int32 )

        candidate = numpy.concatenate( [ self.positive ] * n_copy + [ disjoint, overlap ] )

        if shuffle_needed:
            numpy.random.shuffle( candidate )
        else:
            candidate.sort()
        n = len(candidate)


        for i in range( n ):
            next_example = self.example[ candidate[i] ]
            begin_idx = next_example.begin_idx
            end_idx = next_example.end_idx

            sentence = self.sentence1[next_example.sentence_id]

            if self.language != 'cmn':
                phrase = ' '.join( sentence.sentence[begin_idx:end_idx] )
                phrase_array = numpy.asarray(
                    [ ord(c) for c in list(phrase) ],
                    dtype = numpy.int32
                )
                phrase_view = phrase_array

            if feature_choice & 256 > 0:
                gaz_view[cnt][next_example.gazetteer] = 1

            with nogil:
                label.push_back( next_example.label )

                if feature_choice & (512 | 64) > 0:
                    phrase_cpy_len = context_limit * 2 - 2
                    if phrase_view.shape[0] < phrase_cpy_len:
                        phrase_cpy_len = phrase_view.shape[0]
                    lcv[cnt][1: 1 + phrase_cpy_len] = phrase_view[:phrase_cpy_len]
                    if phrase_cpy_len + 2 > clen:
                        clen = phrase_cpy_len + 2

                if feature_choice & 64 > 0:
                    rcv[cnt][1: 1 + phrase_cpy_len] = phrase_view[::-1][:phrase_cpy_len]

            if feature_choice & 1 > 0:
                lw1len = max( lw1len, sentence.insert_left( end_idx - 1, lw1[cnt] ) )
                rw1len = max( rw1len, sentence.insert_right( begin_idx, rw1[cnt] ) )

            if feature_choice & 2 > 0:
                lw2len = max( lw2len, sentence.insert_left( begin_idx - 1, lw2[cnt] ) )
                rw2len = max( rw2len, sentence.insert_right( end_idx, rw2[cnt] ) )

            if feature_choice & 4 > 0:
                bowlen = max( bowlen, sentence.insert_bow( begin_idx, end_idx, bow1[cnt] ) )

            sentence = self.sentence2[next_example.sentence_id]

            if feature_choice & 8 > 0:
                lw3len = max( lw3len, sentence.insert_left( end_idx - 1, lw3[cnt] ) )
                rw3len = max( rw3len, sentence.insert_right( begin_idx, rw3[cnt] ) )

            if feature_choice & 16 > 0:
                lw4len = max( lw4len, sentence.insert_left( begin_idx - 1, lw4[cnt] ) )
                rw4len = max( rw4len, sentence.insert_right( end_idx, rw4[cnt] ) )

            if feature_choice & 32 > 0:
                bowlen = max( bowlen, sentence.insert_bow( begin_idx, end_idx, bow2[cnt] ) )

            cnt += 1
            if cnt % n_batch_size == 0 or (i + 1) == len(candidate):
                yield {
                    'word' : {
                        'case-insensitive' : {
                            'left-incl' : lw1[:cnt,:lw1len],
                            'right-incl' : rw1[:cnt,:rw1len],
                            'left-excl' : lw2[:cnt,:lw2len],
                            'right-excl' : rw2[:cnt,:rw2len],
                            'bow' : bow1[:cnt,:bowlen]
                        },
                        'case-sensitive' : {
                            'left-incl' : lw3[:cnt,:lw3len],
                            'right-incl' : rw3[:cnt,:rw3len],
                            'left-excl' : lw4[:cnt,:lw4len],
                            'right-excl' : rw4[:cnt,:rw4len],
                            'bow' : bow2[:cnt,:bowlen]
                        }
                    },
                    'char' : {
                        'left' : lc[:cnt,:clen],
                        'right' : rc[:cnt,clen]
                    },
                    'gaz' : gaz_buff[:cnt,:],
                    'target' : label[:cnt]
                }

                with nogil:
                    lw1len = 1
                    rw1len = 1
                    lw2len = 1
                    rw2len = 1
                    lw3len = 1
                    rw3len = 1
                    lw4len = 1
                    rw4len = 1
                    bowlen = 1
                    clen = 10
                    label.clear()
                    cnt = 0
                    lw1v[:,:] = -1
                    rw1v[:,:] = -1
                    lw2v[:,:] = -1
                    rw2v[:,:] = -1
                    lw3v[:,:] = -1
                    rw3v[:,:] = -1
                    lw4v[:,:] = -1
                    rw4v[:,:] = -1
                    bow1v[:,:] = -1
                    bow2v[:,:] = -1
                    lcv[:,:] = 127
                    rcv[:,:] = 127


    def __str__( self ):
        """
        Returns
        -------
            Return a string description of this object.
        """
        return ('%d sentences, %d (positive), %d (overlap), %d (disjoint)' % 
                (len(self.sentence1), 
                    self.positive.shape[0], self.overlap.shape[0], self.disjoint.shape[0]) )


