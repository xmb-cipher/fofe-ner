#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python
# -*- coding: utf-8 -*-

import sys, codecs, logging, json, os, argparse, urllib, xmltodict, HTMLParser
from pycorenlp import StanfordCoreNLP
from pprint import pprint, pformat
from copy import deepcopy
from hanziconv import HanziConv
from time import sleep


logger = logging.getLogger( __name__ )


def load_solution( filename ):
    n_fail = 0

    # TODO: reading a json should be much easier than reading an xml.
    # However, when json.loads is used, it throws exception.
    # If the input XML is malformed, this line throws exception.
    # The exception is supossed to be handled by the caller,
    # and the current file should be skipped.
    with codecs.open( filename + '.xml', 'rb', 'utf8' ) as fp:
        data = json.loads( json.dumps( xmltodict.parse( fp.read() ) ) )

    # with codecs.open( filename + '.json', 'rb', 'utf8' ) as fp:
    # with open( filename + '.json' ) as fp:
    #   data = json.loads( fp.read().decode( 'utf8' ) )

    logger.debug( u'\n' + pformat( data, width = 128, indent = 4 ) )

    solution = {}

    if u'entity' in data[u'source_file'][u'document']:
        entities = data[u'source_file'][u'document'][u'entity']
        if not isinstance( entities, list ):
            assert isinstance( entities, dict )
            entities = [ entities ]

        for entity in entities:
            type1 = entity[u'@TYPE']
            # assert type1 in { 'PER', 'GPE', 'LOC', 'ORG', 'FAC', 'TITLE' }, \
            #         u'%s: %s' % (filename, entity)

            mentions = entity[u'entity_mention']
            if not isinstance( entity[u'entity_mention'], list ):
                assert isinstance( mentions, dict )
                mentions = [ mentions ]

            for mention in mentions :
                logger.debug( mention )
                type2 = mention[u'@TYPE']
                # assert type2 in { 'NAME', 'NOMINAL', 'PRO' }, \
                #                 u'%s: %s' % (filename, mention)

                text = HanziConv.toSimplified( mention[u'head'][u'charseq'][u'#text'] )
                begin = int(mention[u'head'][u'charseq'][u'@START'])
                end = int(mention[u'head'][u'charseq'][u'@END']) + 1

                if type1 in { 'PER', 'GPE', 'LOC', 'ORG', 'FAC', 'TITLE' } \
                    and type2 in { 'NAME', 'NOMINAL' }:
                    # try:
                    #     assert len(text) == end - begin
                    # except Exception as ex:
                    #     logger.exception( u'bad label  %s: %s, %d-%d' % \
                    #                       (filename, text, begin, end) )
                    #     n_fail += 1
                    #     sleep( 1 )
                    if len(text) != end - begin:
                        logger.exception( u'bad label  %s: %s, %d-%d' % (filename, text, begin, end) )
                    solution[ (begin, end) ] = [ text, 'DUMMY', type1, type2 ]

                del type2, text, begin, end
            del type1

    return solution, n_fail



def process_one_chunck( parsed, solution, data, position = 0, language = 'eng' ):
    parsed_bracket = { '-LRB-', '-RRB-', '-LSB-', '-RSB-', '-LCB-', '-RCB-' }
    out_str = u''

    for sentence in parsed[u'sentences']:
        extra = u''
        words, offsets = [], []
        for tokens in sentence[u'tokens']:
            if language != 'cmn':
                word = tokens[u'word']
                begin_offset = tokens[u'characterOffsetBegin']
                while True:
                    next_dash = word.find( u'-' )
                    if word in parsed_bracket:
                        next_dash = -1

                    if next_dash != -1:
                        if next_dash != 0:
                            words.append( word[:next_dash] )
                        words.append( u'-' )
                        end_offset = begin_offset + next_dash

                        if next_dash != 0:
                            offsets.append( ( begin_offset + position, 
                                              end_offset + position) )
                        offsets.append( ( end_offset + position, 
                                          end_offset + 1 + position) )

                        word = word[next_dash + 1:]
                        begin_offset += next_dash + 1
                    else:
                        if len(word) > 0:   # in case that dash is the last character
                            words.append( html_parser.unescape( word ) )
                            offsets.append( ( begin_offset + position, 
                                              tokens[u'characterOffsetEnd'] + position ) )
                        break

            else:
                # extra += u'(%d,%d,%s) ' % ( tokens[u'characterOffsetBegin'] + position,
                #                             tokens[u'characterOffsetEnd'] + position,
                #                             tokens[u'word'] ) 

                word = tokens[u'word']
                begin_offset = tokens[u'characterOffsetBegin']
                has_chinese = any( u'\u4e00' <= c <= u'\u9fff' for c in word )
                if has_chinese:
                    for i,c in enumerate( word ):
                        words.append( u'%s|iNCML|%s' % (c, html_parser.unescape(word)) ) 
                        offsets.append( ( begin_offset + position + i,
                                          begin_offset + position + i + 1 ) )
                else:
                    words.append( u'%s|iNCML|%s' % (word, html_parser.unescape(word)) )
                    offsets.append( ( begin_offset + position, 
                                      tokens[u'characterOffsetEnd'] + position ) )

        assert len(words) == len(offsets), '#words & #offsets unmatched'

        for word, offset in zip( words, offsets ):
            if word != data[offset[0]:offset[1]]:
                logger.debug( u'%s, %s, (%d,%d)' % \
                              ( word, data[offset[0]:offset[1]],
                                offset[0], offset[1] ) )

        label = []
        if solution is not None:
            for i in xrange(len(offsets)):
                for j in xrange(i, len(offsets)):
                    candidate = (offsets[i][0], offsets[j][1])
                    joint_str = u''.join( data[candidate[0]:candidate[1]].split() ).lower()
                    if candidate in solution:
                        sol_str = u''.join(solution[candidate][0].split() ).lower()
                        if joint_str == sol_str:
                            label.append( u'(%d,%d,%s,%s,%s)' % \
                                          tuple([i, j + 1] + solution[candidate][1:]) )
                            solution.pop( candidate )
                    else:   # deal with mis-labled offset
                        for low, high in solution.keys():
                            sol_str = u''.join(solution[(low,high)][0].split()).lower()
                            if joint_str ==  sol_str and \
                                    candidate[0] <= high and low <= candidate[1]:   
                                label.append( u'(%d,%d,%s,%s,%s)' % \
                                              tuple([i, j + 1] + solution[(low,high)][1:]) )
                                solution.pop( (low,high) ) 

        out_str += u' '.join( words ) + u'\n' + \
                   u' '.join( u'(%d,%d)' % (b,e) for b,e in offsets ) + u'\n'
        if len(extra) > 0:
            out_str += extra + u'\n' 
        if len(label) > 0:
            out_str += u' '.join( label ) + u'\n'
        out_str += u'\n'

    return out_str




def process_one_file( input_dir, output_dir, filename, solution, language ):

    full_name = os.path.join(input_dir, filename)
    tags = u'\n'

    with codecs.open( full_name, 'rb', 'utf8' ) as fp:
        data = fp.read()
        data = HanziConv.toSimplified( data )

    # JUST TO SEE THE QUALITY OF OFFSETS
    # for b, e in solution:
    #     mention = data[b:e]#.replace( u'\n', u' ' )
    #     gold = solution[(b, e)][0]
    #     logger.debug( u'%s, %s' % (mention, gold) )
    #     # try:
    #     #     assert mention == gold
    #     # except AssertionError as ex:
    #     #     logger.exception( u'incorrect offset: %d-%d, %s, %s\n' \
    #     #                       % (b, e, mention, gold) )
    #     #     n_fail += 1
    #     #     sleep(1)
    #     assert mention == gold,\
    #            u'incorrect offset: %d-%d, %s, %s\n' % (b, e, mention, gold)

    properties = { 'annotators': 'tokenize,ssplit',
                   'outputFormat': 'json' }
    if language == 'cmn':
        properties['customAnnotatorClass.tokenize'] = 'edu.stanford.nlp.pipeline.ChineseSegmenterAnnotator'
        properties['tokenize.model'] = 'edu/stanford/nlp/models/segmenter/chinese/ctb.gz'
        properties['tokenize.sighanCorporaDict'] = 'edu/stanford/nlp/models/segmenter/chinese'
        properties['tokenize.serDictionary'] = 'edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz'
        properties['tokenize.sighanPostProcessing'] = 'true'
        properties['ssplit.boundaryTokenRegex'] = urllib.quote_plus( '[!?]+|[。]|[！？]+' )
    if language == 'spa':
        properties['tokenize.language'] = 'es'

    ################################################################################

    if data.startswith( u'<?xml' ) or data.startswith( u'<?XML' ):
        lsb, rsb = data.find( u'<' ), data.find( u'>' )
        assert rsb > lsb
        result = []
        in_quote = False

        while True:
            assert data[lsb] == u'<'
            assert data[rsb] == u'>'

            tags += u'(%d,%d)'.ljust(16, u' ') % (lsb, rsb + 1) + data[lsb: rsb + 1] + u'\n'
            
            # look for post author
            tag = data[lsb: rsb + 1]

            if tag.find( u'<quote' ) != -1:
                in_quote = True
            if tag.find( u'</quote' ) != -1:
                in_quote = False

            post_offset = tag.find( u'<post' )
            if post_offset != -1:
                author_offset = tag.find( u'author', post_offset )
                begin_offset = tag.find( u'"', author_offset ) + 1
                end_offset = tag.find( u'"', begin_offset )
                candidate = ( begin_offset + lsb, 
                              begin_offset + lsb + \
                                len(tag[begin_offset: end_offset].rstrip()) )
                logger.debug( data[candidate[0]:candidate[1]] )

                if solution is not None:
                    if candidate in solution:
                        tags += u'(%d,%d)' % candidate + \
                                u'(%s,%s,%s)\n' % tuple(solution[candidate][1:])
                        solution.pop( candidate )
                    else:   # deal with mis-labled offset
                        for low, high in solution.keys():
                            if tag[begin_offset: end_offset].lower() == \
                                        solution[(low,high)][0].lower() \
                                        and candidate[0] <= high and low <= candidate[1]:   
                                tags += u'(%d,%d)' % (low, high) + \
                                        u'(%s,%s,%s)\n' % tuple(solution[(low,high)][1:]) 
                                solution.pop( (low,high) )

                # post authors are detected during prepocessing
                # this line is part of the final output
                tags += u'(%d,%d) ' % candidate + data[candidate[0]:candidate[1]] + u'\n'
            tags += u'\n'


            # look for tags
            next_lsb = data.find( u'<', rsb + 1 )
            if next_lsb == -1:
                break
            next_rsb = data.find( u'>', next_lsb )

            position = rsb + 1
            text = data[position: next_lsb]
            text = text.replace(u'/', u' ').replace(u'_', u' ')

            # escape %, CoreNLP might complain
            text = text.replace( u'%', u'%25' )
            # text = urllib.quote( text )

            if len( text.strip() ) > 0:
                if in_quote:
                    lsb, rsb = next_lsb, next_rsb
                    continue

                n_leading_whitespace = len(text) - len(text.lstrip())
                text = text[n_leading_whitespace:]
                position += n_leading_whitespace
            
                logger.debug( text )
                parsed = nlp.annotate(  text, properties = properties )
                assert isinstance( parsed, dict ), 'CoreNLP does not return well-formed json'
                result.append( process_one_chunck( parsed, 
                                                   solution, 
                                                   data, 
                                                   position, 
                                                   language ) )

            lsb, rsb = next_lsb, next_rsb

    # plain text
    else:
        n_leading_whitespace = len(data) - len(data.lstrip()) 
        text = data[n_leading_whitespace:].strip().replace( u'%', u'%25' )
        parsed = nlp.annotate(  text, properties = properties )
        assert isinstance( parsed, dict ), 'CoreNLP does not return well-formed json'
        result = [ process_one_chunck( parsed, 
                                       solution, 
                                       data, 
                                       n_leading_whitespace, 
                                       language ) ]


    non_match = u'\n'
    if solution is not None:
        for key,value in sorted(solution.iteritems(), key = lambda x: x[0] ):
            non_match += u'%s, %s %s %s %s\n' % ( unicode(key), 
                            value[0], value[1], value[2], value[3] )
        if len(solution) > 0:
            non_match = non_match.rstrip()
            logger.info( u'%s non-match ones: ' % filename + non_match )

    to_write = u''.join( result ) + tags + non_match +\
               u'\n\n' + u'=' * 128 + u'\n\n\n' + data

    if filename.endswith( '.txt' ):
            filename = filename[:-4]
    # full_name = os.path.join( output_dir, filename )
    # with codecs.open( full_name, 'wb', 'utf8' ) as out_file:
    #     out_file.write( to_write )
    # logger.info( '%s saved' % filename )
    if len( solution ) < 2:
        full_name = os.path.join( output_dir, filename )
        with codecs.open( full_name, 'wb', 'utf8' ) as out_file:
            out_file.write( to_write )
        logger.info( '%s saved' % filename )
    else:
        logger.info( '%s ignored' % filename )
 
    return len(solution) if solution is not None else 0




def process_all_files( input_dir, output_dir, language = 'eng' ):
    prefix = { 'eng' : 'eng_', 'cmn' : 'ch_' }

    n_fail, n_inconsistent = 0, 0
    files = os.listdir( input_dir )

    for filename in files:
        full_name = os.path.join( input_dir, filename )

        if os.path.isdir( full_name ):
            n_f, n_i = process_all_files( full_name, output_dir, language )
            n_fail += n_f
            n_inconsistent += n_i

        elif full_name.endswith('.txt'):
            basename = full_name[:-4]
            if os.path.isfile( basename + '.json' ) and \
                    os.path.isfile( basename + '.xml' ) and \
                    input_dir.find( prefix[language] ) != -1:
                logger.info( '=' * 64 )
                try:
                    solution, n_i = load_solution( basename )
                    n_inconsistent += n_i

                    n_f = process_one_file( input_dir, output_dir, 
                                            filename, solution, language )
                    n_fail += n_f

                    logger.info( '%s processed, %d inconsistency, %d failure\n' \
                                  % (full_name, n_i, n_f) )
                    del solution, n_i, n_f

                except AssertionError as ex1:
                    raise
                except Exception as ex2:
                    logger.exception( 'fail to process %s\n' % filename )
                    sleep( 1 )

    return n_fail, n_inconsistent



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--language', type = str, default = 'eng', 
                         choices = [ 'eng', 'cmn' ]  )
    args = parser.parse_args()


    # set a logging file at DEBUG level, TODO: windows doesn't allow ":" appear in a file name
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level= logging.INFO,
                         filename = 'iflytek-parser.log', 
                         filemode = 'w' )

    # direct the INFO-level logging to the screen
    console = logging.StreamHandler()
    console.setLevel( logging.INFO )
    console.setFormatter( logging.Formatter( '%(asctime)s : %(levelname)s : %(message)s' ) )
    logging.getLogger().addHandler( console )

    for handler in logging.root.handlers:
        handler.addFilter( logging.Filter(__name__) )

    # setup CoreNLP
    url = 'http://localhost:2054'
    nlp = StanfordCoreNLP( url )

    html_parser = HTMLParser.HTMLParser()

    n_fail, n_inconsistent = process_all_files( 'iflytek-dataset/checked', 
                                                'iflytek-clean-%s/checked' % args.language,
                                                args.language )
    n_fail, n_inconsistent = process_all_files( 'iflytek-dataset/unchecked', 
                                                'iflytek-clean-%s/unchecked' % args.language,
                                                args.language )
    logger.info( '%d labels are incosistent; fail to process %d mentions' % \
                 (n_inconsistent, n_fail) )
