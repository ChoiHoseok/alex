from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import numpy as np
import input_data

tf.set_random_seed(777)  # reproducibility
NUM_CLASSES = 10
train_x, _, train_y = input_data.load_training_data()
test_x, _, test_y = input_data.load_test_data()
# hyper parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100

TOWER_NAME = 'tower'

def next_batch(i, batch_size):
    batch_num = int(50000 / batch_size)
    if(i > batch_num - 1):
        while(i > batch_num):
            i -= batch_num
        if(i < 0):
            i = 0
    batch_x = train_x[i * batch_size:(i + 1) * batch_size]
    batch_y = train_y[i * batch_size:(i + 1) * batch_size]
    return batch_x, batch_y

def test_batch():
    x_test = test_x[0:batch_size]
    y_test = test_y[0:batch_size]
    return x_test, y_test

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16
        var = tf.get_variable(name, shape, initializer=initializer, dtype = dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float16
    var = _variable_on_cpu(name, shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        self.X = tf.placeholder(tf.float16, [batch_size, 32, 32, 3])
        self.Y = tf.placeholder(tf.float16, [None, 10])
        self.training = tf.placeholder(tf.bool)
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay('weights',
                             shape =[5, 5, 3, 64], stddev=5e-2,wd=None)
            conv = tf.nn.conv2d(self.X, kernel, [1, 1, 1, 1],
                                padding='SAME')
            biases = _variable_on_cpu('biases', [64],
                                        tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            #_activation_summary(conv1)

        pool1 = tf.nn.max_pool(conv1, ksize = [1, 3, 3, 1],
                strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool1')

        norm1 = tf.nn.lrn(pool1, 4, bias = 1.0, alpha = 0.001 / 9.0,
                                            beta = 0.75, name = 'norm1')

        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay('weights',
                             shape =[5, 5, 64, 64], stddev=5e-2,wd=None)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1],
                                padding='SAME')
            biases = _variable_on_cpu('biases', [64],
                                        tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            #_activation_summary(conv2)

        pool2 = tf.nn.max_pool(conv2, ksize = [1, 3, 3, 1],
                strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool2')

        norm2 = tf.nn.lrn(pool2, 4, bias = 1.0, alpha = 0.001 / 9.0,
                                            beta = 0.75, name = 'norm2')
        with tf.variable_scope('local3') as scope:
            reshape = tf.reshape(norm2, [batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay('weights', shape=[dim,384],
                                                stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            #_activation_summary(local3)

        with tf.variable_scope('local4') as scope:
            weights = _variable_with_weight_decay('weights', shape =[384, 192],
                                                stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3,weights) + biases, name=scope.name)
            #_activation_summary(local4)

        with tf.variable_scope('softmax_linear') as scope:
            weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                                stddev=1/192.0, wd = None)
            biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                        tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name = scope.name)
        labels = tf.cast(self.Y, tf.int64)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=softmax_linear)
        self.cost = tf.reduce_mean(cross_entropy, name='cross_entropy')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})
# initialize
#with tf.Session() as sess:
sess = tf.Session()
m1 = Model(sess, "m1")
#merged = tf.summary.merge_all()
#train_writer = tf.summary.FileWriter('/tmp/logs/train2',sess.graph)
sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(50000 / batch_size)

    for i in range(total_batch):
        batch_x, batch_y = next_batch(i, batch_size)
        c, _= m1.train(batch_x, batch_y)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
#train_writer.add_summary(summary)
print('Learning Finished!')

# Test model and check accuracy
X_t,Y_t = test_batch()
print('Accuracy:', m1.get_accuracy(X_t, Y_t))
