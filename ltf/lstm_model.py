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
        embedded_input = tf.nn.embedding_lookup(embeddings, _inputs)

    with tf.variable_scope("lstm"):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_layer_size)
        # outputs - are just our h(t) of shape (batch_size, T, hidden_dim)
        # state - h(T) for the whole batch; shape is not clear
        #
        # so why do we need 2 outputs? why not use outputs[:, -1, :]?
        # it's not possible vor variable length strings!
        # https://stackoverflow.com/a/42850963/2047442
        # https://stackoverflow.com/a/44163122/2047442
        outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                           inputs=embedded_input,
                                           sequence_length=_seqlens,
                                           dtype=tf.float32)

        ################ construct y_hat for last output ################
        weights = {'linear_layer': tf.Variable(
            tf.truncated_normal([hidden_layer_size, num_classes], mean=0, stddev=.01))}
        biases = {'linear_layer': tf.Variable(
            tf.truncated_normal([num_classes], mean=0, stddev=.01))}
        last_output = state[1]
        # extract the last relevant output and use in a linear layer
        scores_for_last_output = tf.matmul(last_output, weights["linear_layer"]) + biases["linear_layer"]

        ################ loss function and optimizer ####################
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=_labels, logits=scores_for_last_output))
        train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

        # it seems we have manually compute accuracy
        correct_prediction = tf.equal(tf.argmax(_labels, 1),
                                      tf.argmax(scores_for_last_output, 1))
        accuracy = (tf.reduce_mean(tf.cast(correct_prediction,
                                           tf.float32))) * 100

        return _inputs, _labels, _seqlens, train_step, accuracy


if __name__ == '__main__':
    train_x, train_y, train_seqlens, test_x, test_y, test_seqlens, word2index_map = generate_data()
    _inputs, _labels, _seqlens, train_step, accuracy = get_lstm_model(vocabulary_size=len(word2index_map))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(300):
            x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size,
                                                                train_x, train_y,
                                                                train_seqlens,
                                                                word2index_map)
            sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch,
                                            _seqlens: seqlen_batch})

            if step % 100 == 0:
                acc = sess.run(accuracy, feed_dict={_inputs: x_batch,
                                                    _labels: y_batch,
                                                    _seqlens: seqlen_batch})
                print("Accuracy at %d: %.5f" % (step, acc))
