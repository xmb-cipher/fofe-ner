#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import logging, argparse, numpy, codecs
logger = logging.getLogger()

if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level= logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument( 'basename', type = str, help = 'basename.{-word.word2vec,-word.wordlist}' )
    args = parser.parse_args()

    logger.info( args )

    with open( '%s-word.word2vec' % args.basename, 'rb' ) as in_file:
    	shape = numpy.fromfile( in_file, dtype = numpy.int32, count = 2 )
    	word2vec = numpy.fromfile( in_file, dtype = numpy.float32 ).reshape( shape )

    with codecs.open( '%s-word.wordlist' % args.basename, 'rb', 'utf8' ) as in_file, \
    	 codecs.open( '%s-avg.wordlist' % args.basename, 'wb', 'utf8' ) as out_file:
    	data = in_file.read()
    	wordlist = [ w.strip() for w in data.strip().split( u'\n' ) ]
    	out_file.write( data )

    assert len(wordlist) == word2vec.shape[0], '%d, %d' % (len(wordlist), word2vec.shape[0])

    for i in xrange( len(wordlist) ):
    	has_chinese = any( u'\u4e00' <= c <= u'\u9fff' for c in wordlist[i] )
    	if has_chinese:
    		word2vec[i] /= len( wordlist[i] )

    with open( '%s-avg.word2vec' % args.basename, 'wb' ) as out_file:
    	numpy.int32( word2vec.shape[0] ).tofile( out_file )
    	numpy.int32( word2vec.shape[1] ).tofile( out_file )
    	word2vec.tofile( out_file )
