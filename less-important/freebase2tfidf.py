#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import argparse, cPickle, numpy, logging
from scipy.sparse import csr_matrix
from itertools import groupby
from sklearn.feature_extraction.text import TfidfTransformer

logger = logging.getLogger()



class SparseVector( object ):
    def __init__( self, indices, values ):
        assert( len(indices) == len(values) )
        self.indices = indices 
        self.values = values 

    def __str__( self ):
        return 'nnz: %d; indices: %s; values: %s' % \
                ( len(self.indices), str(self.indices), str(self.values) )


if __name__ == '__main__':
    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                         level = logging.INFO )

    parser = argparse.ArgumentParser()
    parser.add_argument( '--data', type = str ) #, default = 'toy-input' )
    parser.add_argument( '--word_list', type = str )#, 
                         # default = '/eecs/research/asr/mingbin/ner-advance/' + \
                         #           'word2vec/gigaword256-case-insensitive.wordlist' )
    parser.add_argument( '--output_base', type = str, default = 'tfidf'  )

    args = parser.parse_args()

    result_indices, result_data, result_indptr = [], [], [0]

    with open( args.word_list ) as fp:
        idx2word = fp.read().split()
        word2idx = { w:i for (i, w) in enumerate( idx2word ) }

    int2mid, mid2int, unk = [], {}, word2idx['<unk>']
    with open( args.data ) as fp:
        def convert( x ):
            x = x.rsplit(',', 1)
            assert len(x) == 2, 'x = %s' % str(x)
            return word2idx.get(x[0].lower(), unk), int(x[1])

        fp.readline()   # skip first line
        for line in fp:
            tokens = line.split()
            assert len(tokens) == 2

            head = tokens[0][1:-1]
            head = head[head.rfind('m/'):]
            assert head not in mid2int

            tail = filter( lambda x : x[0] != unk, 
                           map( convert, tokens[1][1:-1].split('};{') ) )
            tail.sort( key = lambda x : x[0] )
            tail = [ ( x[0], sum(y for _,y in x[1]) ) for x in groupby( tail, key = lambda x : x[0] )]

            indices = [ numpy.int32(i) for i, _ in tail ]
            values = [ numpy.float32(v) for _,v in tail ]

            logger.info( SparseVector( indices, values  ) )

            mid2int[head] = len(mid2int)
            int2mid.append( head )

            result_indices.extend( indices )
            result_data.extend( values )
            result_indptr.append( result_indptr[-1] + len(values) )

    counts = csr_matrix( (result_data, result_indices, result_indptr),
                          shape = [len(result_indptr) - 1, len(word2idx)] )
    del result_data, result_indices, result_indptr

    transformer = TfidfTransformer( sublinear_tf = True )
    tfidf = transformer.fit_transform( counts )
    del counts

    tfidf.indices.astype( numpy.int32 ).tofile( args.output_base + '.indices' )
    tfidf.data.astype( numpy.float32 ).tofile( args.output_base + '.data' )
    tfidf.indptr.astype( numpy.int32 ).tofile( args.output_base + '.indptr' )
    numpy.asarray( int2mid ).tofile( args.output_base + '.midlist', sep = '\n' )
