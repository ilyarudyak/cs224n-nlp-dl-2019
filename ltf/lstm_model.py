import numpy as np
import tensorflow as tf

from data_utils import generate_data, get_sentence_batch

batch_size = 128
embedding_dimension = 64
num_classes = 2
hidden_layer_size = 32
times_steps = 6
element_size = 1


def get_lstm_model(vocabulary_size=10):
    _inputs = tf.placeholder(tf.int32, shape=[batch_size, times_steps])
    _labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
    # seqlens for dynamic calculation
    _seqlens = tf.placeholder(tf.int32, shape=[batch_size])

    # it looks like we're going to train our embeddings
    # see the link below for description of tf.nn.embedding_lookup() -
    # basically it just retrieves rows from our embedding matrix of shape (V, n)
    # where V - vocab_size, n - embed_size
    # https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do
    #
    with tf.name_scope("embeddings"):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size,
                               embedding_dimension],
                              -1.0, 1.0), name='embedding')
        embed = tf.nn.embedding_lookup(embeddings, _inputs)
