#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import os, codecs, numpy, argparse, logging
logger = logging.getLogger()


def count_char_freq( filename, freq ):
    if os.path.isdir( filename ):
        smallest, largest = 2 ** 32, -2 ** 32
        for f in os.listdir( filename ):
            fullname = os.path.join( filename, f )
            count_char_freq( fullname, freq )

    else:
        with codecs.open( filename, 'rb', 'utf8' ) as fp:
            data = fp.read()
        data = [ ord(c) for c in data ]
        for c in data:
            freq[c] += 1
        logger.info( '%s processed' % filename )



if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level= logging.INFO)

    parser = argparse.ArgumentParser( description = 'count character frequency of Spanish' )
    parser.add_argument( 'rspecifier', type = str, 
                         help = 'a text file or a directory containing text files' )
    parser.add_argument( 'wspecifier', type = str, 
                         help = 'one unicode per line, sorted by frequency' )

    args = parser.parse_args()
    logger.info( args )

    freq = numpy.zeros( 65536, dtype = numpy.int64 )
    count_char_freq( args.rspecifier, freq )

    result = '\n'.join( str(i) for i in freq.argsort()[::-1] )
    with open( args.wspecifier, 'wb' ) as fp:
        fp.write( result )
