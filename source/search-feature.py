#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

from subprocess import Popen, PIPE, call
import numpy


features = ['case-insensitive bidirectional-context-with-focus', \
			'case-insensitive bidirectional-context-without-focus', \
			'case-insensitive focus-bow', \
			'case-sensitive bidirectional-context-with-focus', \
			'case-sensitive bidirectional-context-without-focus', \
			'case-sensitive focus-bow',\
			'char-level fofe',\
			'char-level initial-fofe',\
			'gazetteer' ]

# feature_test = open( 'log/feature-combination', 'wb', 0 )

# for fc in xrange( 1, 64 ):
# 	print '\nfeature choice: ', [ name for (ith,name) in enumerate(features) if (1 << ith) & fc > 0 ]
# 	print >> feature_test, '\nfeature choice: ', [ name for (ith,name) in enumerate(features) if (1 << ith) & fc > 0 ]

# 	exit_code = call( ("tensorflow ner-trainer-conll2003.py word2vec/reuters256 processed-data " +  
# 						 "--feature_choice=%d --learning_rate=0.1024 --max_iter=12" % fc ).split() ) 

# 	cmd = "python CoNLL2003eval.py processed-data/eng.testa valid.predicted | conlleval"
# 	process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
# 	(out, err) = process.communicate()
# 	exit_code = process.wait()
# 	print out
# 	print >> feature_test, out

# 	cmd = "python CoNLL2003eval.py processed-data/eng.testb test.predicted | conlleval"
# 	process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
# 	(out, err) = process.communicate()
# 	exit_code = process.wait()
# 	print out
# 	print >> feature_test, out

# 	cmd = "python CoNLL2003eval.py processed-data/eng.train train.predicted | conlleval"
# 	process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
# 	(out, err) = process.communicate()
# 	exit_code = process.wait()
# 	print out
# 	print >> feature_test, out

# 	print

# feature_test.close()


################################################################################


# sample_test = open( 'log/sample-rate-combination', 'wb', 0 )

# for disjoint in numpy.arange( 0.1, 1, 0.1 ):
# 	for overlap in numpy.arange( disjoint, 1, 0.1 ):
# 		print '\noverlap-rate: %f,  disjoint-rate: %f' % (overlap, disjoint)
# 		print >> sample_test, '\noverlap-rate: %f,  disjoint-rate: %f' % (overlap, disjoint)

# 		exit_code = call( ("tensorflow ner-trainer-conll2003.py word2vec/reuters256 processed-data " +  
# 							 "--feature_choice=63 --max_iter=12 --learning_rate=0.1024 " + 
# 							 "--overlap_rate=%f --disjoint_rate=%f " % (overlap, disjoint) ).split() ) 

# 		cmd = "python CoNLL2003eval.py processed-data/eng.testa valid.predicted | conlleval"
# 		process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
# 		(out, err) = process.communicate()
# 		exit_code = process.wait()
# 		print out
# 		print >> sample_test, out

# 		cmd = "python CoNLL2003eval.py processed-data/eng.testb test.predicted  | conlleval"
# 		process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
# 		(out, err) = process.communicate()
# 		exit_code = process.wait()
# 		print out
# 		print >> sample_test, out

# 		cmd = "python CoNLL2003eval.py processed-data/eng.train train.predicted | conlleval"
# 		process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
# 		(out, err) = process.communicate()
# 		exit_code = process.wait()
# 		print out
# 		print >> sample_test, out

# 		print

# sample_test.close()


################################################################################



# batch_lr_test = open( 'log/batch-lr-combination', 'wb', 0 )

# for batch in [ 64, 128, 256, 512, 1024 ]:
# 	for lr in [ 0.0256, 0.0512, 0.1024, 0.2048, 0.4096 ]:
# 		print '\nbatch-size: %f,  learning-rate: %f' % (batch, lr)
# 		print >> batch_lr_test, '\nbatch-size: %f,  learning-rate: %f' % (batch, lr)

# 		exit_code = call( ("tensorflow ner-trainer-conll2003.py word2vec/reuters256 processed-data " +  
# 							 "--feature_choice=63 --max_iter=12  " + 
# 							 "--learning_rate=%f --n_batch_size=%d " % (lr, batch) ).split() ) 

# 		cmd = "python CoNLL2003eval.py processed-data/eng.testa valid.predicted | conlleval"
# 		process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
# 		(out, err) = process.communicate()
# 		exit_code = process.wait()
# 		print out
# 		print >> batch_lr_test, out

# 		cmd = "python CoNLL2003eval.py processed-data/eng.testb test.predicted  | conlleval"
# 		process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
# 		(out, err) = process.communicate()
# 		exit_code = process.wait()
# 		print out
# 		print >> batch_lr_test, out

# 		cmd = "python CoNLL2003eval.py processed-data/eng.train train.predicted | conlleval"
# 		process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
# 		(out, err) = process.communicate()
# 		exit_code = process.wait()
# 		print out
# 		print >> batch_lr_test, out

# 		print

# batch_lr_test.close()



# to see the effect of a single feature




embedding_effect = open( 'embedding-effect', 'wb', 0 )

for e in ['reuters128', 'reuters256', 'gigaword128', 'gigaword256']:
	print '\nembedding: ', e
	print >> embedding_effect, '\nfeature choice: ', e

	exit_code = call( ("python ner-trainer-conll2003.py word2vec/%s processed-data " % e +  
						 "--feature_choice=63 --learning_rate=0.128 --max_iter=64 --dropout=True " ).split() ) 

	cmd = "python CoNLL2003eval.py processed-data/eng.testa valid.predicted | conlleval"
	process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
	(out, err) = process.communicate()
	exit_code = process.wait()
	print out
	print >> embedding_effect, out

	cmd = "python CoNLL2003eval.py processed-data/eng.testb test.predicted | conlleval"
	process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
	(out, err) = process.communicate()
	exit_code = process.wait()
	print out
	print >> embedding_effect, out

	cmd = "python CoNLL2003eval.py processed-data/eng.train train.predicted | conlleval"
	process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
	(out, err) = process.communicate()
	exit_code = process.wait()
	print out
	print >> embedding_effect, out

	print

embedding_effect.close()





single_feature = open( 'single-feature', 'wb', 0 )

for i, c in enumerate( [1, 2, 4, 8, 16, 32, 64, 128, 256, 7, 56, 63] ):
	print '\nfeature choice: ', c#, features[i]
	print >> single_feature, '\nfeature choice: ', c#, features[i]

	exit_code = call( ("python ner-trainer-conll2003.py word2vec/reuters256 processed-data " +  
						 "--feature_choice=%d --learning_rate=0.1024 --max_iter=16 --char_alpha=0.8" % c ).split() ) 

	cmd = "python CoNLL2003eval.py processed-data/eng.testa valid.predicted | conlleval"
	process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
	(out, err) = process.communicate()
	exit_code = process.wait()
	print out
	print >> single_feature, out

	cmd = "python CoNLL2003eval.py processed-data/eng.testb test.predicted | conlleval"
	process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
	(out, err) = process.communicate()
	exit_code = process.wait()
	print out
	print >> single_feature, out

	cmd = "python CoNLL2003eval.py processed-data/eng.train train.predicted | conlleval"
	process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
	(out, err) = process.communicate()
	exit_code = process.wait()
	print out
	print >> single_feature, out

	print

single_feature.close()

