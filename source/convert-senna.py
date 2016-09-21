#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import re, argparse, numpy, logging
logger = logging.getLogger()


if __name__ == '__main__':
	logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
						 level= logging.INFO)

	parser = argparse.ArgumentParser()
	parser.add_argument( 'vector', type = str )
	parser.add_argument( 'output', type = str )
	args = parser.parse_args()
	vector = args.vector
	output = args.output

	with open( vector, 'rb' ) as fp:
		word2vec = numpy.asarray( [ [ numpy.float32(f) for f in v.strip().split() ] for v in fp ] )

	fp = open( output, 'wb' )
	numpy.int32(word2vec.shape[0]).tofile( fp )
	numpy.int32(word2vec.shape[1]).tofile( fp )
	word2vec.tofile( fp )
	fp.close()

