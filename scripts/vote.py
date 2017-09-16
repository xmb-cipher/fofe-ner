#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python


import argparse
import os
import codecs
import logging
logger = logging.getLogger()

if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s',
                         level = logging.INFO )

    parser = argparse.ArgumentParser()
    parser.add_argument( 'dir', type = str )
    parser.add_argument( '--run_id', type = int, default = 1 )
    parser.add_argument( '--model_cnt', type = int, default = 5 )
    args = parser.parse_args()

    logger.info( args )


    solution = {}
    for i in xrange( args.model_cnt ):
        filename = os.path.join( args.dir, '%d.tsv' % i )
        with codecs.open( filename, 'rb', 'utf8' ) as fp:
            for line in fp:
                tokens = line.strip().split(u'\t')
                key = (tokens[3], tokens[5], tokens[6])
                if key in solution:
                    solution[key]['score'] = 0.2 + solution[key]['score']
                else:
                    solution[key] = { 'spelling' : tokens[2], 'score' : 0.2 }
        logger.info( '%d.tsv processed' % i )
    logger.info( len(solution) )

    weight = 1. / args.model_cnt
    for i in xrange(args.model_cnt):
        threshold = weight * i
        cnt = 0 
        output = os.path.join( args.dir, '%02d-of-%02d.tsv' % (i, args.model_cnt) )
        with codecs.open( output, 'wb', 'utf8' ) as fp:
            for key in solution:
                if solution[key]['score'] > threshold:
                    position, mention1, mention2 = key
                    out_str = u'\t'.join( [ 
                        u'YorkNRM%d' % args.run_id,  
                        u'TEDL16_EVAL_%06d' % cnt, 
                        solution[key]['spelling'], 
                        position,
                        u'NIL000000', 
                        mention1, 
                        mention2, 
                        u'%.1f' % solution[key]['score']
                    ] )
                    fp.write( out_str + u'\n' )
                    cnt += 1


