#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import os, argparse, codecs, logging

logger = logging.getLogger( __name__ )



if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level = logging.INFO )

    parser = argparse.ArgumentParser()
    parser.add_argument( 'input_dir', type = str ) 
    parser.add_argument( 'output', type = str )
    parser.add_argument( '--language', type = str, default = 'eng',
                         choices = [ 'eng', 'cmn', 'spa' ]  )
    parser.add_argument( '--run_id', type = int, default = 1 )
    parser.add_argument( '--post_author_list', type = str, default = None )

    args = parser.parse_args()
    cnt = 0

    if args.post_author_list:
        post_author_list = codecs.open( args.post_author_list, 'wb', 'utf8' )

    with codecs.open( args.output, 'wb', 'utf8' ) as outfile:
        for filename in os.listdir( args.input_dir ):
            full_path = os.path.join( args.input_dir, filename )
            with codecs.open( full_path, 'rb', 'utf8' ) as fp:
                data = fp.read()
            processed, original = data.split( u'=' * 128, 1 )
            original = original.strip()

            texts, tags, failures = processed.split( u'\n\n\n', 2 )

            for text in texts.split( u'\n\n' ):
                parts = text.split( u'\n' )
                assert len(parts) in [2, 3], 'sentence, offsets, labels(optional)'

                sent = parts[0].split(u' ')
                offsets = map( lambda x : (int(x[0]), int(x[1])),
                               [ offsets[1:-1].split(u',') for offsets in parts[1].split() ] )
                assert len(offsets) == len(sent) 

                if len(parts) == 3:
                    for ans in parts[-1].split():
                        cnt += 1
                        begin_idx, end_idx, _, mention1, mention2 = ans[1:-1].split(u',')
                        begin_idx, end_idx = int(begin_idx), int(end_idx)
                        c_begin, c_end =  offsets[begin_idx][0], offsets[end_idx - 1][1]
                        if args.language != 'cmn':
                            spelling = original[ c_begin: c_end ].replace( u'\n', u' ' )
                        else:
                            spelling = original[ c_begin: c_end ].replace( u'\n', u'' )

                        # TODO: here's a quick fix in order to make the validator happy
                        if not c_begin < c_end:
                            continue

                        out_str = u'\t'.join( [ u'YorkNRM%d' % args.run_id, 
                                                u'TEDL16_EVAL_%06d' % cnt, 
                                                spelling.replace( u'\t', u' ' ), 
                                                u'%s:%d-%d' % (filename, c_begin, c_end - 1),
                                                u'NIL000000', mention1, mention2, u'1.0' ] )
                                                # u'1.0', u'N', u'N', u'N' ] )
                        outfile.write( out_str + u'\n' )

            for tag in tags.rstrip().split( u'\n\n' ):
                parts = tag.split( u'\n' )
                # assert len(parts) in [1, 2, 3], 'tag, human-label, machine-label'
                # if len( parts ) == 3:
                assert len(parts) in [1, 2], 'tag, machine-label'
                if len( parts ) == 2:
                    cnt += 1
                    # offset, spelling = parts[2].split( u' ', 1 )
                    offset, spelling = parts[-1].split( u' ', 1 )
                    c_begin, c_end = [ int(x) for x in offset[1:-1].split( u',' ) ]

                    # TODO: here's a quick fix in order to make the validator happy
                    if not c_begin < c_end:
                        continue

                    out_str = u'\t'.join( [ u'YorkNRM%d' % args.run_id,  
                                            u'TEDL16_EVAL_%06d' % cnt, 
                                            spelling.replace( u'\t', u' ' ), 
                                            u'%s:%d-%d' % (filename, c_begin, c_end - 1),
                                            u'NIL000000', 'PER', 'NAM', u'1.0' ] )
                                            # u'1.0', u'N', u'N', u'N' ] )
                    outfile.write( out_str + u'\n' )

                    if args.post_author_list:
                        post_author_list.write( u'%s:%d-%d\n' % (filename, c_begin, c_end - 1) )

            logger.info( '%s processed' % filename )

    if args.post_author_list:
        post_author_list.close()

