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

    nil_cnt = 0
    for f in os.listdir( args.rspecifier ):
        n_nil = 0
        full_name = os.path.join( args.rspecifier, f )
        with codecs.open( full_name, 'rb', 'utf8' ) as in_file:
            for line in in_file:
                tokens = line.strip().split(u'\t')
                key, score = tokens[3], tokens[7]
                if tokens[4].startswith('NIL'):
                    # linking is done separately, they all start with NIL1
                    # so we have to increment the NIL count.
                    mid = 'NIL%d' % (int(tokens[4][3:]) + nil_cnt)
                    n_nil = max(n_nil, int(tokens[4][3:]))
                else:
                    mid = tokens[4]
                if float(tokens[7]) > 1.0:
                    tokens[7] = '1.0'
                if key not in solution or score > solution[key]['score']:
                    solution[key] = {
                        'spelling' : tokens[2],
                        'mid' : mid,
                        'ed-type1' : tokens[5],
                        'ed-type2' : tokens[6],
                        'score' : tokens[7]
                    }
        nil_cnt += n_nil
        logger.info( '%s processed' % f )


    with codecs.open( args.wspecifier, 'wb', 'utf8' ) as out_file:
        for cnt, key in enumerate( sorted(solution.keys()) ):
            ans = solution[key]
            out_str = u'\t'.join( [ 
                u'YorkNRM%d' % args.run_id,  
                u'TEDL17_EVAL_%06d' % cnt, 
                ans['spelling'], 
                key,
                ans['mid'], 
                ans['ed-type1'], 
                ans['ed-type2'], 
                ans['score']
            ] )
            out_file.write( out_str + u'\n' )
