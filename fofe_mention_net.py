#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

"""
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : fofe_mention_net.py
Last Update : Jul 17, 2016
Description : N/A
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
"""


import numpy, logging, time, copy, os, cPickle

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# from CoNLL2003eval import evaluation
from gigaword2feature import * 
from LinkingUtil import *

from tqdm import tqdm
from itertools import ifilter, izip, imap
from random import choice

logger = logging.getLogger( __name__ )



########################################################################


def load_word_embedding( filename ):
    """
    Parameters
    ----------
        filename : str
            path to the word embedding binary file (trained by skipgram-train.py)

    Returns
    -------
        embedding : ndarray
            2D matrix where each row is a word vector
    """
    with open( filename, 'rb' ) as fp:
        shape = numpy.fromfile( fp, dtype = numpy.int32, count = 2 )
        embedding = numpy.fromfile( fp, dtype = numpy.float32 ).reshape( shape )
    return embedding


########################################################################


class mention_config( object ):
    def __init__( self, args = None ):
        # default config
        self.word_embedding = 'word2vec/reuters128'
        self.data_path = 'processed-data'
        self.n_char_embedding = 64
        self.n_char = 128
        self.n_batch_size = 512
        self.learning_rate = 0.1024
        self.momentum = 0.9
        self.layer_size = '512,512,512'
        self.max_iter = 64
        self.feature_choice = 511
        self.overlap_rate = 0.08
        self.disjoint_rate = 0.016
        self.dropout = True
        self.n_ner_embedding = 32
        self.char_alpha = 0.8
        self.word_alpha = 0.5
        self.n_window = 7
        self.strictly_one_hot = True
        self.hope_out = 0
        self.n_label_type = 7
        self.kernel_height = range(2, 10)
        self.kernel_depth = [16] * 8
        self.enable_distant_supervision = False
        self.initialize_method = 'uniform'
        self.kernel_depth = ','.join( ['16'] * 8 )
        self.kernel_height = '2,3,4,5,6,7,8,9'
        self.l1 = 0
        self.l2 = 0

        # KBP-specific config
        self.language = 'eng'
        self.average = False
        self.is_2nd_pass = False

        if args is not None:
            self.__dict__.update( args.__dict__ )

        self.kernel_depth = [ int(d) for d in self.kernel_depth.split(',') ]
        self.kernel_height = [ int(h) for h in self.kernel_height.split(',') ]
        
        # these parameters are not decided by the input to the program
        # I put some placeholders here; they will be eventually modified
        self.algorithm = 1              # highest first, decided by training
        self.threshold = 0.5            # decided by training
        self.drop_rate = 0.4096 if self.dropout else 0
        self.n_word1 = 100000           # decided by self.word_embedding
        self.n_word2 = 100000           # decided by self.word_embedding
        self.n_word_embedding1 = 256    # decided by self.word_embedding
        self.n_word_embedding2 = 256    # decided by self.word_embedding
        self.customized_threshold = None    # not used any more
        assert len( self.kernel_height ) == len( self.kernel_depth )


########################################################################


class fofe_mention_net( object ):
    def __init__( self, config = None, gpu_option = 0.96 ):
        """
        Parameters
        ----------
            config : mention_config
        """

        # most code is lengacy, let's put some alias here
        word_embedding = config.word_embedding
        data_path = config.data_path
        n_char_embedding = config.n_char_embedding
        n_char = config.n_char
        n_batch_size = config.n_batch_size
        learning_rate = config.learning_rate
        momentum = config.momentum
        layer_size = config.layer_size
        feature_choice = config.feature_choice
        overlap_rate = config.overlap_rate
        disjoint_rate = config.disjoint_rate
        dropout = config.dropout
        n_ner_embedding = config.n_ner_embedding
        char_alpha = config.char_alpha
        word_alpha = config.word_alpha
        n_window = config.n_window
        hope_out = config.hope_out
        n_label_type = config.n_label_type
        kernel_height = config.kernel_height
        kernel_depth = config.kernel_depth
        enable_distant_supervision = config.enable_distant_supervision
        initialize_method = config.initialize_method
 
        if config is not None:
            self.config = copy.deepcopy( config )
        else:
            self.config = mention_config()

        self.graph = tf.Graph()
        # TODO: create a graph instead of using default graph
        #       otherwise, we cannot instantiate multiple fofe_mention_nets
        # tf.reset_default_graph()
        if gpu_option is not None:
            gpu_option = tf.GPUOptions( per_process_gpu_memory_fraction = gpu_option )
            self.session = tf.Session( config = tf.ConfigProto( gpu_options = gpu_option ),
                                       graph = self.graph )
        else:
             self.session = tf.Session( graph = self.graph )

        if os.path.exists( self.config.word_embedding + '-case-insensitive.word2vec' ) \
            and os.path.exists( self.config.word_embedding + '-case-sensitive.word2vec' ):

            projection1 = load_word_embedding( self.config.word_embedding + \
                                               '-case-insensitive.word2vec' )
            projection2 = load_word_embedding( self.config.word_embedding + \
                                               '-case-sensitive.word2vec' )

            self.n_word1 = projection1.shape[0]
            self.n_word2 = projection2.shape[0]

            n_word_embedding1 = projection1.shape[1]
            n_word_embedding2 = projection2.shape[1]

            self.config.n_word1 = self.n_word1
            self.config.n_word2 = self.n_word2
            self.config.n_word_embedding1 = n_word_embedding1
            self.config.n_word_embedding2 = n_word_embedding2
            logger.info( 'non-Chinese embeddings loaded' )

        elif os.path.exists( self.config.word_embedding + '-char.word2vec' ) \
            and os.path.exists( self.config.word_embedding + '-word.word2vec' ):

            projection1 = load_word_embedding( self.config.word_embedding + \
                                               '-char.word2vec' )
            projection2 = load_word_embedding( self.config.word_embedding + \
                            ('-avg.word2vec' if self.config.average else '-word.word2vec') )

            self.n_word1 = projection1.shape[0]
            self.n_word2 = projection2.shape[0]

            n_word_embedding1 = projection1.shape[1]
            n_word_embedding2 = projection2.shape[1]

            self.config.n_word1 = self.n_word1
            self.config.n_word2 = self.n_word2
            self.config.n_word_embedding1 = n_word_embedding1
            self.config.n_word_embedding2 = n_word_embedding2
            logger.info( 'Chinese embeddings loaded' )

        else:
            self.n_word1 = self.config.n_word1
            self.n_word2 = self.config.n_word2
            n_word_embedding1 = self.config.n_word_embedding1
            n_word_embedding2 = self.config.n_word_embedding2

            projection1 = numpy.random.uniform( -1, 1, 
                            (self.n_word1, n_word_embedding1) ).astype( numpy.float32 )
            projection2 = numpy.random.uniform( -1, 1, 
                            (self.n_word2, n_word_embedding2) ).astype( numpy.float32 )
            logger.info( 'embedding is randomly initialized' )

        if config.is_2nd_pass:
            logger.info( 'In 2nd pass, substitute the last few entries with label types.' )
            projection1[-1 - n_label_type: -1, :] = \
                            numpy.random.uniform( 
                                    projection1.min(), projection1.max(),
                                    (n_label_type, n_word_embedding1) 
                            ).astype( numpy.float32 )
            sub = numpy.random.uniform( 
                                projection2.min(), projection2.max(),
                                (n_label_type, n_word_embedding2) 
                        ).astype( numpy.float32 )
            if self.config.language == 'cmn':
                projection2[-1 - n_label_type: -1, :] = sub
            else:
                projection2[-2 - n_label_type: -2, :] = sub

        # dimension of x in the HOPE paper
        hope_in = 0
        for ith, name in enumerate( ['case-insensitive bidirectional-context-with-focus', \
                                     'case-insensitive bidirectional-context-without-focus', \
                                     'case-insensitive focus-bow', \
                                     'case-sensitive bidirectional-context-with-focus', \
                                     'case-sensitive bidirectional-context-without-focus', \
                                     'case-sensitive focus-bow', \
                                     'left-char & right-char', 'left-initial & right-initial', \
                                     'gazetteer', 'char-conv', 'char-bigram' ] ):
            if (1 << ith) & self.config.feature_choice > 0: 
                logger.info( '%s used' % name )
                if ith in [0, 1]:
                    hope_in += n_word_embedding1 * 2
                elif ith in [3, 4]:
                    hope_in += n_word_embedding2 * 2
                elif ith == 2:
                    hope_in += n_word_embedding1
                elif ith == 5:
                    hope_in += n_word_embedding1
                elif ith in [6, 7]:
                    hope_in += n_char_embedding * 2
                elif ith == 8: 
                    hope_in += n_ner_embedding
                elif ith == 9:
                    hope_in += sum( kernel_depth )
                elif ith == 10:
                    hope_in += n_char_embedding * 2

        # add a U matrix between projected feature and fully-connected layers
        n_in = [ hope_out if hope_out > 0 else hope_in ] + [ int(s) for s in layer_size.split(',') ]

        # output size of fully-connected layers
        n_out = n_in[1:] + [ n_label_type + 1 ]

        logger.info( 'n_in: ' + str(n_in) )
        logger.info( 'n_out: ' + str(n_out) )

        with self.graph.as_default():
            ################################################################################
            #################### placeholder ###############################################
            ################################################################################

            self.lw1_values = tf.placeholder( tf.float32, [None], 
                                              name = 'left-context-values' )
            self.lw1_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'left-context-indices' )

            self.rw1_values = tf.placeholder( tf.float32, [None], 
                                              name = 'right-context-values' )
            self.rw1_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'right-context-indices' )

            self.lw2_values = tf.placeholder( tf.float32, [None], 
                                              name = 'left-context-values' )
            self.lw2_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'left-context-indices' )

            self.rw2_values = tf.placeholder( tf.float32, [None], 
                                              name = 'right-context-values' )
            self.rw2_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'right-context-indices' )

            self.bow1_values = tf.placeholder( tf.float32, [None], 
                                               name = 'bow-values' )
            self.bow1_indices = tf.placeholder( tf.int64, [None, 2], 
                                                name = 'bow-indices' )

            self.lw3_values = tf.placeholder( tf.float32, [None], 
                                              name = 'left-context-values' )
            self.lw3_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'left-context-indices' )

            self.rw3_values = tf.placeholder( tf.float32, [None], 
                                              name = 'right-context-values' )
            self.rw3_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'right-context-indices' )

            self.lw4_values = tf.placeholder( tf.float32, [None], 
                                              name = 'left-context-values' )
            self.lw4_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'left-context-indices' )

            self.rw4_values = tf.placeholder( tf.float32, [None], 
                                              name = 'right-context-values' )
            self.rw4_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'right-context-indices' )

            self.bow2_values = tf.placeholder( tf.float32, [None], 
                                               name = 'bow-values' )
            self.bow2_indices = tf.placeholder( tf.int64, [None, 2], 
                                                name = 'bow-indices' )

            self.shape1 = tf.placeholder( tf.int64, [2], name = 'bow-shape1' )
            self.shape2 = tf.placeholder( tf.int64, [2], name = 'bow-shape2' )

            self.lc_fofe = tf.placeholder( tf.float32, [None, n_char], name = 'left-char' )
            self.rc_fofe = tf.placeholder( tf.float32, [None, n_char], name = 'right-char' )

            self.li_fofe = tf.placeholder( tf.float32, [None, n_char], name = 'left-initial' )
            self.ri_fofe = tf.placeholder( tf.float32, [None, n_char], name = 'right-initial' )

            self.ner_cls_match = tf.placeholder( tf.float32, [None, n_label_type + 1], name = 'gazetteer' )

            self.label = tf.placeholder( tf.int64, [None], 'label' )

            self.lr = tf.placeholder( tf.float32, [], 'learning-rate' )

            self.keep_prob = tf.placeholder( tf.float32, [], 'keep-prob' )
            
            self.char_idx = tf.placeholder( tf.int32, [None, None], name = 'char-idx' )

            self.lbc_values = tf.placeholder( tf.float32, [None], name = 'bigram-values' )
            self.lbc_indices = tf.placeholder( tf.int64, [None, 2], name = 'bigram-indices' )

            self.rbc_values = tf.placeholder( tf.float32, [None], name = 'bigram-values' )
            self.rbc_indices = tf.placeholder( tf.int64, [None, 2], name = 'bigram-indices' )

            self.shape3 = tf.placeholder( tf.int64, [2], name = 'shape3' )

            logger.info( 'placeholder defined' )

            ################################################################################
            #################### model parameters ##########################################
            ################################################################################

            self.word_embedding_1 = tf.Variable( projection1 )
            self.word_embedding_2 = tf.Variable( projection2 )
            del projection1, projection2

            self.W = []
            self.b = []   # weights & bias of fully-connected layers
            self.param = []

            if initialize_method == 'uniform':
                val_rng = numpy.float32(2.5 / numpy.sqrt(n_char + n_char_embedding))
                self.char_embedding = tf.Variable( tf.random_uniform( 
                                        [n_char, n_char_embedding], minval = -val_rng, maxval = val_rng ) )
            
                self.conv_embedding = tf.Variable( tf.random_uniform( 
                                        [n_char, n_char_embedding], minval = -val_rng, maxval = val_rng ) )

                val_rng = numpy.float32(2.5 / numpy.sqrt(n_label_type + n_ner_embedding + 1))
                self.ner_embedding = tf.Variable( tf.random_uniform( 
                                        [n_label_type + 1 ,n_ner_embedding], minval = -val_rng, maxval = val_rng ) )
                
                val_rng = numpy.float32(2.5 / numpy.sqrt(96 * 96 + n_char_embedding))
                self.bigram_embedding = tf.Variable( tf.random_uniform( 
                                        [96 * 96, n_char_embedding], minval = -val_rng, maxval = val_rng ) )
                
                self.kernels = [ tf.Variable( tf.random_uniform( 
                                    [h, n_char_embedding, 1, d], 
                                    minval = -2.5 / numpy.sqrt(1 + h * n_char_embedding * d), 
                                    maxval = 2.5 / numpy.sqrt(1 + h * n_char_embedding * d) ) ) for \
                                (h, d) in zip( kernel_height, kernel_depth ) ]

                self.kernel_bias = [ tf.Variable( tf.zeros( [d] ) ) for d in kernel_depth ]

                if hope_out > 0:
                    val_rng = 2.5 / numpy.sqrt( hope_in + hope_out )
                    u_matrix = numpy.random.uniform( -val_rng, val_rng, 
                                                    [hope_in, hope_out] ).astype( numpy.float32 )
                    u_matrix = u_matrix / (u_matrix ** 2).sum( 0 )
                    self.U = tf.Variable( u_matrix )
                    del u_matrix

                for i, o in zip( n_in, n_out ):
                    val_rng = numpy.float32(2.5 / numpy.sqrt(i + o))
                    self.W.append( tf.Variable( tf.random_uniform( [i, o], minval = -val_rng, maxval = val_rng ) ) )
                    self.b.append( tf.Variable( tf.zeros( [o] ) )  )
                  
                del val_rng
                
            else:
                self.char_embedding = tf.Variable( tf.truncated_normal( [n_char, n_char_embedding], 
                                                stddev = numpy.sqrt(2./(n_char * n_char_embedding)) ) )
                
                self.conv_embedding = tf.Variable( tf.truncated_normal( [n_char, n_char_embedding], 
                                                stddev = numpy.sqrt(2./(n_char * n_char_embedding)) ) )

                self.ner_embedding = tf.Variable( tf.truncated_normal( [n_label_type + 1, n_ner_embedding], 
                                                stddev = numpy.sqrt(2./(n_label_type * n_ner_embedding)) ) )

                self.bigram_embedding = tf.Variable( tf.truncated_normal( [96 * 96, n_char_embedding],
                                                stddev = numpy.sqrt(2./(96 * 96 * n_char_embedding)) ) )

                self.kernels = [ tf.Variable( tf.truncated_normal( [h, n_char_embedding, 1, d], 
                                                              stddev = numpy.sqrt(2./(h * n_char_embedding * d)) ) ) for \
                            (h, d) in zip( kernel_height, kernel_depth ) ]

                self.kernel_bias = [ tf.Variable( tf.zeros( [d] ) ) for d in kernel_depth ]

                # the U matrix in the HOPE paper
                if hope_out > 0:
                    self.U = tf.Variable( tf.truncated_normal( [hope_in, hope_out],
                                                          stddev = numpy.sqrt(2./(hope_in * hope_out)) ) )

                for i, o in zip( n_in, n_out ):
                    self.W.append( tf.Variable( tf.truncated_normal( [i, o], stddev = numpy.sqrt(2./(i * o)) ) ) )
                    self.b.append( tf.Variable( tf.zeros( [o] ) ) )

            if hope_out > 0:
                self.param.append( self.U )
            self.param.append( self.char_embedding )
            self.param.append( self.conv_embedding )
            self.param.append( self.ner_embedding )
            self.param.append( self.bigram_embedding )
            self.param.extend( self.kernels )
            self.param.extend( self.kernel_bias )
            self.param.extend( self.W )
            self.param.extend( self.b )
            logger.info( 'variable defined' )

            ################################################################################

            char_cube = tf.expand_dims( tf.gather( self.conv_embedding, self.char_idx ), 3 )
            char_conv = [ tf.reduce_max( tf.nn.tanh( tf.nn.conv2d( char_cube, 
                                                                   kk, 
                                                                   [1, 1, 1, 1], 
                                                                   'VALID' ) + bb ),
                                                     reduction_indices = [1, 2] ) \
                            for kk,bb in zip( self.kernels, self.kernel_bias) ]

            lw1 = tf.SparseTensor( self.lw1_indices, self.lw1_values, self.shape1 )
            rw1 = tf.SparseTensor( self.rw1_indices, self.rw1_values, self.shape1 )
            lw2 = tf.SparseTensor( self.lw2_indices, self.lw2_values, self.shape1 )
            rw2 = tf.SparseTensor( self.rw2_indices, self.rw2_values, self.shape1 )
            bow1 = tf.SparseTensor( self.bow1_indices, self.bow1_values, self.shape1 )

            lw3 = tf.SparseTensor( self.lw3_indices, self.lw3_values, self.shape2 )
            rw3 = tf.SparseTensor( self.rw3_indices, self.rw3_values, self.shape2 )
            lw4 = tf.SparseTensor( self.lw4_indices, self.lw4_values, self.shape2 )
            rw4 = tf.SparseTensor( self.rw4_indices, self.rw4_values, self.shape2 )
            bow2 = tf.SparseTensor( self.bow2_indices, self.bow2_values, self.shape2 )

            lbc = tf.SparseTensor( self.lbc_indices, self.lbc_values, self.shape3 )
            rbc = tf.SparseTensor( self.rbc_indices, self.rbc_values, self.shape3 )

            # all sparse feature after projection
            lwp1 = tf.sparse_tensor_dense_matmul( lw1, self.word_embedding_1 )
            rwp1 = tf.sparse_tensor_dense_matmul( rw1, self.word_embedding_1 )
            lwp2 = tf.sparse_tensor_dense_matmul( lw2, self.word_embedding_1 )
            rwp2 = tf.sparse_tensor_dense_matmul( rw2, self.word_embedding_1 )
            bowp1 = tf.sparse_tensor_dense_matmul( bow1, self.word_embedding_1 )
            lwp3 = tf.sparse_tensor_dense_matmul( lw3, self.word_embedding_2 )
            rwp3 = tf.sparse_tensor_dense_matmul( rw3, self.word_embedding_2 )
            lwp4 = tf.sparse_tensor_dense_matmul( lw4, self.word_embedding_2 )
            rwp4 = tf.sparse_tensor_dense_matmul( rw4, self.word_embedding_2 )
            bowp2 = tf.sparse_tensor_dense_matmul( bow2, self.word_embedding_2 )

            # dense features after projection
            lcp = tf.matmul( self.lc_fofe, self.char_embedding )
            rcp = tf.matmul( self.rc_fofe, self.char_embedding )

            lip = tf.matmul( self.li_fofe, self.char_embedding )
            rip = tf.matmul( self.ri_fofe, self.char_embedding )

            lbcp = tf.sparse_tensor_dense_matmul( lbc, self.bigram_embedding )
            rbcp = tf.sparse_tensor_dense_matmul( rbc, self.bigram_embedding )

            ner_projection = tf.matmul( self.ner_cls_match, self.ner_embedding )

            # all possible features
            feature_list = [ [lwp1, rwp1], [lwp2, rwp2], [bowp1],
                             [lwp3, rwp3], [lwp4, rwp4], [bowp2],
                             [lcp, rcp], [lip, rip], [ner_projection],
                             char_conv, [lbcp, rbcp] ]
            used, not_used = [], [] 

            # decide what feature to use
            for ith, f in enumerate( feature_list ):
                if (1 << ith) & feature_choice > 0: 
                    used.extend( f )
                else:
                    not_used.extend( f )
            feature_list = used #+ not_used

            # feature = tf.concat( 1, feature_list )
            feature = tf.concat( feature_list, 1 )

            # if hope is used, add one linear layer
            if hope_out > 0:
                hope = tf.matmul( feature, U )
                layer_output = [ hope ] 
            else:
                layer_output = [ tf.nn.dropout( feature, self.keep_prob ) ]

            for i in xrange( len(self.W) ):
                layer_output.append( tf.matmul( layer_output[-1], self.W[i] ) + self.b[i] )
                if i < len(self.W) - 1:
                    layer_output[-1] = tf.nn.relu( layer_output[-1] )
                    layer_output[-1] = tf.nn.dropout( layer_output[-1], self.keep_prob )

            self.xent = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( 
                                            logits = layer_output[-1], labels = self.label ) )

            if config.l1 > 0:
                for param in self.param:
                    self.xent = self.xent + config.l1 * tf.reduce_sum( tf.abs( param ) )

            if config.l2 > 0:
                for param in  self.param:
                    self.xent = self.xent + config.l2 * tf.nn.l2_loss( param )

            self.predicted_values = tf.nn.softmax( layer_output[-1] )
            _, top_indices = tf.nn.top_k( self.predicted_values )
            self.predicted_indices = tf.reshape( top_indices, [-1] )

            # fully connected layers are must-trained layers
            fully_connected_train_step = tf.train.MomentumOptimizer( self.lr, 
                                                                     self.config.momentum, 
                                                                     use_locking = False ) \
                                           .minimize( self.xent, var_list = self.W + self.b )
            self.train_step = [ fully_connected_train_step ]


            if feature_choice & 0b111 > 0:
                insensitive_train_step = tf.train.GradientDescentOptimizer( self.lr / 4, 
                                                                            use_locking = True ) \
                                          .minimize( self.xent, var_list = [ self.word_embedding_1 ] )
                self.train_step.append( insensitive_train_step )

            if feature_choice & (0b111 << 3) > 0:
                sensitive_train_step = tf.train.GradientDescentOptimizer( self.lr / 4, 
                                                                          use_locking = True ) \
                                          .minimize( self.xent, var_list = [ self.word_embedding_2 ] )
                self.train_step.append( sensitive_train_step )


            if feature_choice & (0b11 << 6) > 0:
                char_embedding_train_step = tf.train.GradientDescentOptimizer( self.lr / 2, 
                                                                               use_locking = True ) \
                                              .minimize( self.xent, var_list = [ self.char_embedding ] )
                self.train_step.append( char_embedding_train_step )

            if feature_choice & (1 << 8) > 0:
                ner_embedding_train_step = tf.train.GradientDescentOptimizer( self.lr, 
                                                                              use_locking = True ) \
                                          .minimize( self.xent, var_list = [ self.ner_embedding ] )
                self.train_step.append( ner_embedding_train_step )

            if feature_choice & (1 << 9) > 0:
                char_conv_train_step = tf.train.MomentumOptimizer( self.lr, momentum )\
                                             .minimize( self.xent, 
                                                var_list = [ self.conv_embedding ] + \
                                                             self.kernels + self.kernel_bias )
                self.train_step.append( char_conv_train_step )

            if feature_choice & (1 << 10) > 0:
                bigram_train_step = tf.train.GradientDescentOptimizer( self.lr / 2, use_locking = True )\
                                            .minimize( self.xent, var_list = [ self.bigram_embedding ] )
                self.train_step.append( bigram_train_step )

            if hope_out > 0:
                ui_norm = tf.sqrt( tf.reduce_sum( U ** 2, 
                                                  reduction_indices = [ 0 ],
                                                  keep_dims = True  )  )        # 1 x hope_out
                ui_norm_dot_uj_norm = tf.matmul( ui_norm, 
                                                 ui_norm, 
                                                 transpose_a = True )           # hope_out x hope_out
                ui_dot_uj = tf.matmul( U, U, transpose_a = True ) 
                orthogonal_penalty = tf.reduce_sum( tf.mul( tf.abs( ui_dot_uj ) / ui_norm_dot_uj_norm,
                                                            1 - tf.diag( tf.ones( [ hope_out ] ) ) ) )
                orthogonal_penalty = orthogonal_penalty / hope_out

                U_train_step_1 = tf.train.GradientDescentOptimizer( lr, use_locking = True ) \
                                         .minimize( orthogonal_penalty , var_list = [ self.U ] )
                U_train_step_2 = tf.train.MomentumOptimizer( lr, momentum, use_locking = True ) \
                                         .minimize( xent, var_list = [ self.U ] )
                train_step.append( U_train_step_1 )
                train_step.append( U_train_step_2 )

                self.normalize_step = self.U.assign( tf.nn.l2_normalize( self.U, 0 ) )

        logger.info( 'computational graph built\n' )

        with self.graph.as_default():
            self.session.run( tf.global_variables_initializer() )
            # self.session.run( tf.variables_initializer( self.param ) )
            self.saver = tf.train.Saver()



    def train( self, mini_batch ):
        """
        Parameters
        ----------
            mini_batch : tuple

        Returns
        -------
            c : float
        """ 
        l1_values, r1_values, l1_indices, r1_indices, \
        l2_values, r2_values, l2_indices, r2_indices, \
        bow1i, \
        l3_values, r3_values, l3_indices, r3_indices, \
        l4_values, r4_values, l4_indices, r4_indices, \
        bow2i, \
        dense_feature,\
        conv_idx,\
        l5_values, l5_indices, r5_values, r5_indices, \
        target = mini_batch

        if not self.config.strictly_one_hot:
            dense_feature[:,-1] = 0  

        c = self.session.run(   self.train_step + [ self.xent ],
                                feed_dict = {   self.lw1_values: l1_values,
                                                self.lw1_indices: l1_indices,
                                                self.rw1_values: r1_values,
                                                self.rw1_indices: r1_indices,
                                                self.lw2_values: l2_values,
                                                self.lw2_indices: l2_indices,
                                                self.rw2_values: r2_values,
                                                self.rw2_indices: r2_indices,
                                                self.bow1_indices: bow1i,
                                                self.bow1_values: numpy.ones( bow1i.shape[0], dtype = numpy.float32 ),
                                                self.lw3_values: l3_values,
                                                self.lw3_indices: l3_indices,
                                                self.rw3_values: r3_values,
                                                self.rw3_indices: r3_indices,
                                                self.lw4_values: l4_values,
                                                self.lw4_indices: l4_indices,
                                                self.rw4_values: r4_values,
                                                self.rw4_indices: r4_indices,
                                                self.bow2_indices: bow2i,
                                                self.bow2_values: numpy.ones( bow2i.shape[0], dtype = numpy.float32 ),
                                                self.shape1: (target.shape[0], self.n_word1),
                                                self.shape2: (target.shape[0], self.n_word2),
                                                self.lc_fofe: dense_feature[:,:128],
                                                self.rc_fofe: dense_feature[:,128:256],
                                                self.li_fofe: dense_feature[:,256:384],
                                                self.ri_fofe: dense_feature[:,384:512],
                                                self.ner_cls_match: dense_feature[:,512:],
                                                self.char_idx: conv_idx,
                                                self.lbc_values : l5_values,
                                                self.lbc_indices : l5_indices,
                                                self.rbc_values : r5_values,
                                                self.rbc_indices : r5_indices,
                                                self.shape3 : (target.shape[0], 96 * 96),
                                                self.label: target,
                                                self.lr: self.config.learning_rate,
                                                self.keep_prob: 1 - self.config.drop_rate } )[-1]

        if self.config.hope_out > 0:
            self.session.run( self.normalize_step )

        return c



    def eval( self, mini_batch ):
        """
        Parameters
        ----------
            mini_batch : tuple

        Returns:
            c : float
            pi : numpy.ndarray
            pv : numpy.ndarray
        """
        l1_values, r1_values, l1_indices, r1_indices, \
        l2_values, r2_values, l2_indices, r2_indices, \
        bow1i, \
        l3_values, r3_values, l3_indices, r3_indices, \
        l4_values, r4_values, l4_indices, r4_indices, \
        bow2i, \
        dense_feature,\
        conv_idx,\
        l5_values, l5_indices, r5_values, r5_indices, \
        target = mini_batch

        if not self.config.strictly_one_hot:
            dense_feature[:,-1] = 0

        c, pi, pv = self.session.run( [ self.xent, 
                                        self.predicted_indices, 
                                        self.predicted_values ], 
                                        feed_dict = {   self.lw1_values: l1_values,
                                                        self.lw1_indices: l1_indices,
                                                        self.rw1_values: r1_values,
                                                        self.rw1_indices: r1_indices,
                                                        self.lw2_values: l2_values,
                                                        self.lw2_indices: l2_indices,
                                                        self.rw2_values: r2_values,
                                                        self.rw2_indices: r2_indices,
                                                        self.bow1_indices: bow1i,
                                                        self.bow1_values: numpy.ones( bow1i.shape[0], dtype = numpy.float32 ),
                                                        self.lw3_values: l3_values,
                                                        self.lw3_indices: l3_indices,
                                                        self.rw3_values: r3_values,
                                                        self.rw3_indices: r3_indices,
                                                        self.lw4_values: l4_values,
                                                        self.lw4_indices: l4_indices,
                                                        self.rw4_values: r4_values,
                                                        self.rw4_indices: r4_indices,
                                                        self.bow2_indices: bow2i,
                                                        self.bow2_values: numpy.ones( bow2i.shape[0], dtype = numpy.float32 ),
                                                        self.shape1: (target.shape[0], self.n_word1),
                                                        self.shape2: (target.shape[0], self.n_word2),
                                                        self.lc_fofe: dense_feature[:,:128],
                                                        self.rc_fofe: dense_feature[:,128:256],
                                                        self.li_fofe: dense_feature[:,256:384],
                                                        self.ri_fofe: dense_feature[:,384:512],
                                                        self.ner_cls_match: dense_feature[:,512:],
                                                        self.char_idx: conv_idx,
                                                        self.lbc_values : l5_values,
                                                        self.lbc_indices : l5_indices,
                                                        self.rbc_values : r5_values,
                                                        self.rbc_indices : r5_indices,
                                                        self.shape3 : (target.shape[0], 96 * 96),
                                                        self.label: target,
                                                        self.keep_prob: 1 } ) 

        return c, pi, pv


    def tofile( self, filename ):
        """
        Parameters
        ----------
            filename : str
                The current model will be stored in basename.{tf,config}
        """
        self.saver.save( self.session, filename )
        with open( filename + '.config', 'wb' ) as fp:
            cPickle.dump( self.config, fp )


    def fromfile( self, filename ):
        """
            filename : str
                The current model will be restored from basename.{tf,config}
        """
        self.saver.restore( self.session, filename )


    def __del__( self ):
        self.session.close()



########################################################################

