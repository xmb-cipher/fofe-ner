#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python
# -*- coding: utf-8 -*-

import sys, codecs, logging, json, os, argparse, urllib, time, HTMLParser
from pycorenlp import StanfordCoreNLP
from pprint import pprint, pformat
from copy import deepcopy
from hanziconv import HanziConv

logger = logging.getLogger(__name__)


def process_one_file( input_dir, output_dir, filename, solution, language, sentence_only = False ):
    # send the text to CoreNLP for annotation
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


    parsed_bracket = { '-LRB-', '-RRB-', '-LSB-', '-RSB-', '-LCB-', '-RCB-' }

    with codecs.open( os.path.join( input_dir, filename), 
                          'rb', 'utf-8' ) as fp:
        data = fp.read()
    # filename = filename.rsplit('.', 1)[0]

    # KBP2015
    if filename[-7:] == '.df.xml' or filename[-7:] == '.nw.xml':
        filename = filename[:-7]
    elif filename[-4:] == '.xml':
        filename = filename[:-4]

    # KBP2016
    elif filename[-3:] == '.df' or filename[-3:] == '.nw':
        filename = filename[:-3]

    out_file = codecs.open( os.path.join( output_dir, filename ), 
                            'wb', 'utf-8' )

    lsb = data.find( u'<DOC' )
    if lsb == -1:
        lsb = data.find( u'<doc' )
    assert lsb >= 0, 'kbp files should start with <DOC> tag'
    data = data[lsb:]
    logger.debug( '<DOC> or <doc> starts at index %d' % lsb )

    assert data.find( u'<' ) == 0, 'Begin of document is incorrect'
    lsb = 0
    rsb = data.find( u'>', lsb )
    assert rsb != -1, 'xml tag doesn not match'

    tags = u'\n'
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

            post_au = tag[begin_offset: end_offset]
            n_leading = len(post_au) - len(post_au.lstrip())   

            candidate = ( begin_offset + lsb + n_leading, 
                          begin_offset + lsb + \
                            len(tag[begin_offset: end_offset].rstrip()) )         

            logger.debug( data[candidate[0]:candidate[1]] )

            if filename in solution:
                if candidate in solution[filename]:
                    tags += u'(%d,%d)' % candidate + \
                            u'(%s,%s,%s)\n' % tuple(solution[filename][candidate][1:])
                    solution[filename].pop( candidate )
                else:   # deal with mis-labled offset
                    for low, high in solution[filename].keys():
                        if tag[begin_offset: end_offset] == \
                                    solution[filename][(low,high)][0] \
                                    and candidate[0] <= high and low <= candidate[1]:   
                            tags += u'(%d,%d)' % (low, high) + \
                                    u'(%s,%s,%s)\n' % tuple(solution[filename][(low,high)][1:]) 
                            solution[filename].pop( (low,high) )

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
        text = text.replace(u'/', u' ')#.replace(u'-', u' ')

        # escape %, CoreNLP might complain
        text = text.replace( u'%', u'%25' )
        # text = urllib.quote( text )

        if len( text.strip() ) > 0:
            # check whether the text under consideration is in quote region
            is_in_quote = False
            if filename in quote:
                for begin, end in quote[filename]:
                    if begin <= position < next_lsb <= end:
                        is_in_quote = True
                        break
                    
            if is_in_quote or in_quote:
                lsb, rsb = next_lsb, next_rsb
                continue

            n_leading_whitespace = len(text) - len(text.lstrip())
            text = text[n_leading_whitespace:]
            position += n_leading_whitespace
        
            logger.debug( text )

            try:
                time.sleep( 0.0128 )
                if language == 'cmn':
                    text = HanziConv.toSimplified( text ) 
                parsed = nlp.annotate(  text, properties = properties )

                assert isinstance( parsed, dict ), 'CoreNLP does not return well-formed json'

                for sentence in parsed[u'sentences']:
                    # words = [ tokens[u'word'] for tokens in sentence[u'tokens'] ]
                    # offsets = [ ( tokens[u'characterOffsetBegin'] + position, \
                    #             tokens[u'characterOffsetEnd'] + position ) \
                    #                           for tokens in sentence[u'tokens'] ]
                    extra = u''
                    words, offsets = [], []
                    for tokens in sentence[u'tokens']:
                        if language != 'cmn':
                            word = tokens[u'word']
                            begin_offset = tokens[u'characterOffsetBegin']
                            print tokens[u'word'], tokens[u'characterOffsetBegin'], tokens[u'characterOffsetEnd']
                            while True:
                                next_dash = word.find( u'-' )

                                if word in parsed_bracket:
                                    next_dash = -1

                                if next_dash != -1:
                                    # print word, next_dash

                                    if next_dash != 0:
                                        words.append( html_parser.unescape( word[:next_dash] ) )
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
                            if not sentence_only:
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
                            else:
                                words.append( html_parser.unescape(word) )

                    for i in xrange(len(words)):
                        words[i] = words[i].replace( u' ', u'\0' )

                    assert sentence_only or len(words) == len(offsets)

                    for word, offset in zip( words, offsets ):
                        if word != data[offset[0]:offset[1]]:
                            logger.debug( u'%s, %s, (%d,%d)' % \
                                          ( word, data[offset[0]:offset[1]],
                                            offset[0], offset[1] ) )

                    label = []
                    if filename in solution:
                        for i in xrange(len(offsets)):
                            for j in xrange(i, len(offsets)):
                                candidate = (offsets[i][0], offsets[j][1])
                                joint_str = u''.join( data[candidate[0]:candidate[1]].split() ).lower()
                                # logger.debug( u'%s, %d, %d' % (joint_str, candidate[0], candidate[1]) )
                                if candidate in solution[filename]:
                                    sol_str = u''.join(solution[filename][candidate][0].split() ).lower()
                                    if joint_str == sol_str:
                                        label.append( u'(%d,%d,%s,%s,%s)' % \
                                                      tuple([i, j + 1] + solution[filename][candidate][1:]) )
                                        solution[filename].pop( candidate )
                                else:   # deal with mis-labled offset
                                    for low, high in solution[filename].keys():
                                        sol_str = u''.join(solution[filename][(low,high)][0].split()).lower()
                                        if joint_str ==  sol_str and \
                                                candidate[0] <= high and low <= candidate[1]:   
                                            label.append( u'(%d,%d,%s,%s,%s)' % \
                                                          tuple([i, j + 1] + solution[filename][(low,high)][1:]) )
                                            solution[filename].pop( (low,high) )                        

                    if sentence_only:
                        out_file.write( u' '.join(words) + u'\n' )
                    else:
                        out_file.write( u' '.join( words ) + u'\n' + \
                                        u' '.join( u'(%d,%d)' % (b,e) for b,e in offsets ) + u'\n' )
                        # if len(extra) > 0:
                        #     out_file.write( extra + u'\n' ) 
                        if len(label) > 0:
                            out_file.write( u' '.join( label ) + u'\n' )
                        out_file.write( u'\n' )

            except:
                logger.exception( filename )
                logger.exception( text )

        lsb, rsb = next_lsb, next_rsb

    if not sentence_only:
        out_file.write( tags )

    non_match = u'\n'
    if filename in solution:
        for key,value in sorted(solution[filename].iteritems(), key = lambda x: x[0] ):
            non_match += u'%s, %s %s %s %s\n' % ( unicode(key), 
                            value[0], value[1], value[2], value[3] )
        out_file.write( non_match )

        if len(solution[filename]) > 0:
            # logger.info( u'%s non-match ones: \n' % filename + \
            #               pformat( solution[filename], width = 128 ) )
            logger.info( u'%s non-match ones: ' % filename + non_match )

    if not sentence_only:
        out_file.write( u'\n\n' + u'=' * 128 + u'\n\n\n' + data )
    out_file.close()

    logger.info( '%s processed\n' % filename )

    # if len( solution[filename] ) > 0:
    #   exit(0)

    return len( solution[filename] ) if filename in solution else 0



def process_all_files( input_dir, output_dir, solution, language, sentence_only = False ):
    n_fail = 0
    files = os.listdir( input_dir )
    for filename in files:
        full_name = os.path.join( input_dir, filename )
        if os.path.isdir( full_name ):
            n_fail += process_all_files( full_name, output_dir, solution, language, sentence_only )
        else:
            n_fail += process_one_file( input_dir, output_dir, filename, solution, language, sentence_only )
    return n_fail





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( 'input_dir', type = str, 
                         help = 'directory containing kbp source xml' )
    parser.add_argument( 'output_dir', type = str,
                         help = 'directory containing parsed sentences' )
    parser.add_argument( '--mention', type = str, default = None,
                         help = 'tac_kbp_2015_tedl_training_gold_standard_entity_mentions.tab' )
    parser.add_argument( '--quote', type = str, default = None,
                         help = 'quote_regions.tab' )
    parser.add_argument( '--verbose', action = 'store_true', default = False )
    parser.add_argument( '--language', type = str, default = 'eng',
                         choices = [ 'eng', 'cmn', 'spa' ] )
    parser.add_argument( '--url', type = str, default = 'http://localhost:2054' )
    parser.add_argument( '--sentence_only', action = 'store_true', default = False )
    args = parser.parse_args()

    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level = logging.DEBUG if args.verbose else logging.INFO )
    
    for handler in logging.root.handlers:
        handler.addFilter( logging.Filter(__name__) )

    logger.info( args )

    # load the solutions
    solution, correct, incorrect = {}, 0, 0
    gold_tab = args.mention
    if gold_tab is not None:
        incorrect_info = u'\n'
        with codecs.open( gold_tab, 'rb', 'utf-8' ) as fp:
            for line in fp:
                # traditional chinese is converted to simplified chinese
                # CJK space is converted to ascii space
                tokens = HanziConv.toSimplified( line ).split( u'\t' )
                if len(tokens) > 6:
                    filename, offset = tokens[3].split(u':')
                    begin_offset, end_offset = [ int(x) for x in offset.split(u'-') ]
                    if filename not in solution:
                        solution[filename] = {}
                    solution[filename][(begin_offset, end_offset + 1)] = [tokens[2], tokens[4], tokens[5], tokens[6]]
                    if end_offset + 1 - begin_offset != len(tokens[2]):
                        incorrect_info += u'%s  %s  %d %d\n' % \
                                          ( filename, tokens[2], begin_offset, end_offset + 1 )
                        incorrect += 1
                    else:
                        correct += 1
            logger.debug( incorrect_info )
            logger.info( '%d correct, %d incorrect\n' % (correct, incorrect) )
                    
    # load the quote regions
    quote = {}
    quote_tab = args.quote
    if quote_tab is not None:
        with codecs.open( quote_tab, 'rb', 'utf-8' ) as fp:
            for line in fp:
                filename, begin_offset, end_offset = line.strip().split(u'\t')
                filename = filename.split(u'.')[0]
                begin_offset, end_offset = int(begin_offset), int(end_offset) + 1
                if filename not in quote:
                    quote[filename] = set() 
                quote[filename].add( (begin_offset,end_offset) )    

    
    # setup CoreNLP
    nlp = StanfordCoreNLP( args.url ) 

    html_parser = HTMLParser.HTMLParser()

    n_fail = process_all_files( args.input_dir, args.output_dir, solution, args.language, args.sentence_only )
    logger.info( u'fail to parse: %d ' % n_fail )
