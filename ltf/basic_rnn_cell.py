# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:34:43 2016
@author: tomhope
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/mnist", one_hot=True)

##################### (1) define params #####################

# that's our image represented as a sequence of vectors each of size 28
element_size = 28
time_steps = 28

num_classes = 10
batch_size = 128

# each of RNN cells is of this size
hidden_layer_size = 128

##################### (2) define input data #####################

_inputs = tf.placeholder(tf.float32,
                         shape=[None, time_steps, element_size],
                         name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='inputs')

##################### (3) define our layers #####################

##################### (3a) RNN layer ############################
# TensorFlow built-in functions
# rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_layer_size)
outputs, _ = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

##################### (3b) linear layer to get y_hat ############
Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],
                                     mean=0, stddev=.01))
bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))


def get_linear_layer(vector):
    return tf.matmul(vector, Wl) + bl


last_rnn_output = outputs[:, -1, :]
final_output = get_linear_layer(last_rnn_output)

##################### (4) define our loss (incl. softmax) ########
softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_output, labels=y)
cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

# it seems we compute accuracy manually
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100


if __name__ == '__main__':
    # sess = tf.InteractiveSession()
    with tf.Session() as sess:
        # (5a) initialize variables
        sess.run(tf.global_variables_initializer())

        # (5b) get test data
        test_data = mnist.test.images[:batch_size].reshape(
            (-1, time_steps, element_size))
        test_label = mnist.test.labels[:batch_size]

        # (5c) our main loop
        for i in range(3001):
            # (5b) get train data
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape((batch_size, time_steps, element_size))
            sess.run(train_step, feed_dict={_inputs: batch_x, y: batch_y})

            if i % 1000 == 0:
                acc = sess.run(accuracy, feed_dict={_inputs: batch_x,
                                                    y: batch_y})
                loss = sess.run(cross_entropy, feed_dict={_inputs: batch_x,
                                                          y: batch_y})
                print("Iter " + str(i) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc))

        print("Testing Accuracy:", sess.run(accuracy,
                                            feed_dict={_inputs: test_data, y: test_label}))
