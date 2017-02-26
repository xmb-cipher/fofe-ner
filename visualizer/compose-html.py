#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import codecs
import pprint
import argparse
import logging
logger = logging.getLogger( __name__ )


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
                    sentence, ner_begin, ner_end, ner_label, last_ner = [], [], [], [], 4



def CoNLL2003Doc( filename ):
    result = []
    for sent, boe, eoe, coe in CoNLL2003( filename ):
        if len(sent) == 1 and sent[0] == '-DOCSTART-':
            if len(result) > 0:
                yield result
                result = []
            continue
        result.append( (sent, boe, eoe, coe) )
    if len(result) > 0:
        yield result



# experimental, inefficient implementation
# should count words one by one
def doc2js( doc ):
    cls2ner = [ 'PER', 'LOC', 'ORG', 'MISC' ]
    text, entities, offset, n_entities = '', [], 0, 0
    for sent, boe, eoe, coe in doc:
        acc_len = [ offset ]
        for w in sent:
            acc_len.append( acc_len[-1] + len(w) + 1 )

        for i in xrange(len(coe)):
            entities.append( [ 'T%d' % n_entities,
                               cls2ner[coe[i]],
                               [[ acc_len[boe[i]], acc_len[eoe[i]] - 1 ]] ] )
            n_entities += 1

        text += u' '.join( sent ) + u'\n'
        offset = acc_len[-1]

    return { 'text': text.encode('ascii', 'ignore'), 'entities': entities }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( 'gold', type = str, help = 'eng.testb' )
    parser.add_argument( 'system', type = str, help = 'output of fofe-ner' )
    parser.add_argument( 'html', type = str, help = 'visualized html' )
    args = parser.parse_args()

    head = \
"""
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html>

<head>
    <meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <title>Minimal Analysis of CoNLL2003</title>
    <link rel="stylesheet" type="text/css" href="css/style-vis.css">
    <script type="text/javascript" src="js/head.js"></script>
</head>

<body>

<!-- load all the libraries upfront, which takes forever. -->
<script type="text/javascript" src="js/brat_loader.js"></script>


<!-- some simple css for parallel div -->
<style type="text/css">
    .left-side {
        float: left;
        width: 48%;
        margin: 10px 2px 10px 16px;
    }

    .right-side {
        float: right;
        width: 48%;
        margin: 10px 16px 10px 2px;
    }

    .center {
        width: 96%;
        margin: 32px 16px 32px 16px;
    }
</style>


<!-- 4 label types -->
<script type="text/javascript">
var collData = {
    entity_types: [ 
    {   type   : 'PER',
        labels : ['PER'],
        bgColor: '#ff0000',
        borderColor: 'darken' },
    {   type   : 'LOC',
        labels : ['LOC'],
        bgColor: '#00ff00',
        borderColor: 'darken' },
    {   type   : 'ORG',
        labels : ['ORG'],
        bgColor: '#0000ff',
        borderColor: 'darken' },
    {   type   : 'MISC',
        labels : ['MISC'],
        bgColor: '#888888',
        borderColor: 'darken' }
    ]
};
</script>



"""

    tail = \
"""
</body>
</html>
"""


    func_head = \
"""
<script type="text/javascript">
    head.ready(function() {
"""

    func_tail = \
"""
    });
</script>
"""

    js1 = '<script type="text/javascript">\n'
    div = ''
    js2 = ''

    for i, (x, y) in enumerate( zip(CoNLL2003Doc(args.gold), CoNLL2003Doc(args.system)) ):
        if x == y:
            continue

        js1 += 'var dataX%d = %s;\n' % (i, str(doc2js(x)))
        js1 += 'var dataY%d = %s;\n' % (i, str(doc2js(y)))

        div += "<div>"        
        div += "<div id = 'docX-%d', class = 'left-side'></div>\n" % i
        div += "<div id = 'docY-%d', class = 'right-side'></div>\n" % i
        div += "</div>"

        js2 += "        Util.embed('docX-%d', collData, dataX%d, webFontURLs);\n" % (i, i)
        js2 += "        Util.embed('docY-%d', collData, dataY%d, webFontURLs);\n" % (i, i)

    js1 += '</script>'
    js2 = '\n'.join([func_head, js2, func_tail])
    
    with open( args.html, 'wb' ) as fp:
        print >> fp, head
        print >> fp, js1
        print >> fp, div
        print >> fp, js2
        print >> fp, tail


        # print >> fp, head
        # print >> fp, '<script type="text/javascript">'

        # itr = CoNLL2003Doc( 'eng.testb' )

        # print >> fp, 'var data%d = %s;' % ( 0, str(doc2js(itr.next())) )
        # # pprint.pprint( doc2js( itr.next() ), fp )

        # print >> fp, "</script>"
        # print >> fp, "<div id = 'doc-%d'></div>" % 0


        # print >> fp, func_head
        # print >> fp, "        Util.embed('doc-%d', collData, data%d, webFontURLs);" % (0, 0)

        # print >> fp, func_tail
        # print >> fp, tail

