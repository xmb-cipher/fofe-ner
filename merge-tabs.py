#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python


import codecs, argparse, logging, os


if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level = logging.INFO )

    parser = argparse.ArgumentParser()
    parser.add_argument( 'rspecifier', type = str, help = 'directory containing all the tab files' )
    parser.add_argument( 'wspecifier', type = str, help = 'path to the output tab file' )
    parser.add_argument( '--run_id', type = int, default = 1 )

    args = parser.parse_args()
    logger.info( str(args) + '\n' ) 

    solution = {}

    cnt, duplicate, disagree = 0, 0, 0
    with codecs.open( args.wspecifier, 'wb', 'utf8' ) as out_file:
        for f in os.listdir( args.rspecifier ):
            full_name = os.path.join( args.rspecifier, f )
            with codecs.open( full_name, 'rb', 'utf8' ) as in_file:
                for line in in_file:
                    tokens = line.split( u'\t' )
                    if len(tokens) == 8:
                        if tokens[3] not in solution:
                            solution[ tokens[3] ] = (tokens[5], tokens[6])
                            cnt += 1
                            tokens[0] = u'YorkNRM%d' % args.run_id
                            tokens[1] = u'TEDL16_EVAL_%08d' % cnt
                            out_file.write( u'\t'.join( tokens ) )
                        else:
                            duplicate += 1
                            if solution[ tokens[3] ] != (tokens[5], tokens[6]):
                                disagree += 1
            logger.info( '%s processed' % f )
        logger.info( '%d duplicate, %d disagree' % (duplicate, disagree) )
