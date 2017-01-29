#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import logging, codecs, random, os, argparse
logger = logging.getLogger()


if __name__ == '__main__':

    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level = logging.INFO )

    parser = argparse.ArgumentParser()
    parser.add_argument( 'in_path', type = str, 
                          help = 'directory containing eng.{train,testa,testb}' )
    parser.add_argument( 'out_path', type = str,
                          help = 'directory containing split-{1,2,3,4,5}' )

    args = parser.parse_args()
    logger.info( str(args) + '\n' )

    train = [ codecs.open( os.path.join(args.out_path, 'split-%d' % i, 'eng.train'),
                         'wb', 'utf8' ) for i in xrange(5) ]
    testa = [ codecs.open( os.path.join(args.out_path, 'split-%d' % i, 'eng.testa'),
                         'wb', 'utf8' ) for i in xrange(5) ]
    opt = range(5)

    for f in [ 'eng.train', 'eng.testa' ]:
        with codecs.open( os.path.join(args.in_path, f),
                          'rb', 'utf8' ) as src:
            data = src.read().strip().split( u'\n\n' )
        for sentence in data:
            rnd = random.choice(opt)
            for i in xrange(5):
                if i == rnd:
                    testa[i].write( u'%s\n\n' % sentence )
                else:
                    train[i].write( u'%s\n\n' % sentence )
        logger.info( '%s processed.' % f )

    for o in train + testa:
        o.close()

