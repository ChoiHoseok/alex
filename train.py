import tensorflow as tf
import numpy as np
import input_data

tf.set_random_seed(777)  # reproducibility
is_convert = False
train_x, _, train_y = input_data.load_training_data(is_convert)
test_x, _, test_y = input_data.load_test_data(is_convert)
is_convert = True
train_convert_x, _, train_convert_y = input_data.load_training_data(is_convert)
# hyper parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 100
#is_train = True
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

def next_converted_batch(i, batch_size):
    batch_num = int(50000 / batch_size)
    if(i > batch_num - 1):
        while(i > batch_num):
            i -= batch_num
        if(i < 0):
            i = 0
    batch_x = train_convert_x[i * batch_size:(i + 1) * batch_size]
    batch_y = train_y[i * batch_size:(i + 1) * batch_size]
    return batch_x, batch_y

def test_batch():
    x_test = test_x[0:batch_size]
    y_test = test_y[0:batch_size]
    return x_test, y_test

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):

            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [batch_size,32,32,3])

            X_img = tf.reshape(self.X, [-1, 32, 32, 3])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            bn1 = tf.layers.batch_normalization(X_img, training=self.training)
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=bn1, filters=32, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=0.7, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            bn2 = tf.layers.batch_normalization(dropout1, training=self.training)
            conv2 = tf.layers.conv2d(inputs=bn2, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.7, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            bn3 = tf.layers.batch_normalization(dropout2, training=self.training)
            conv3 = tf.layers.conv2d(inputs=bn3, filters=128, kernel_size=[3, 3],
                                     padding="same", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=0.7, training=self.training)

            bn4 = tf.layers.batch_normalization(dropout3, training=self.training)
            conv4 = tf.layers.conv2d(inputs=bn4, filters=256, kernel_size=[3, 3],
                                     padding="same", activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout4 = tf.layers.dropout(inputs=pool4,
                                         rate=0.7, training=self.training)

            bn5 = tf.layers.batch_normalization(dropout4, training=self.training)
            conv5 = tf.layers.conv2d(inputs=bn5, filters=256, kernel_size=[3, 3],
                                     padding="same", activation=tf.nn.relu)
            pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout5 = tf.layers.dropout(inputs=pool5,
                                         rate=0.7, training=self.training)

            # Dense Layer with Relu
            bn6 = tf.layers.batch_normalization(dropout4, training=self.training)
            flat = tf.reshape(bn6, [-1, 256 * 4 * 4])
            dense1 = tf.layers.dense(inputs=flat,
                                     units=625, activation=tf.nn.relu)
            dropout6 = tf.layers.dropout(inputs=dense1,
                                         rate=0.5, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout6, units=10)

        #with tf.name_scope('cost'):
            # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        tf.summary.histogram('cost', self.cost)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, merged, training=True):
        return self.sess.run([self.cost, self.optimizer, merged, self.accuracy], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})
# initialize
#with tf.Session() as sess:
sess = tf.Session()
m1 = Model(sess, "m1")
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/tmp/logs',sess.graph)
sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(50000 / batch_size)

    for i in range(total_batch):
        batch_x, batch_y = next_batch(i, batch_size)
        c, _, summary, accuracy = m1.train(batch_x, batch_y, merged)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('Accuracy:', accuracy)
train_writer.add_summary(summary)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(50000 / batch_size)

    for i in range(total_batch):
        batch_convert_x, batch_convert_y = next_converted_batch(i, batch_size)
        c, _, summary, accuracy = m1.train(batch_convert_x, batch_convert_y, merged)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (training_epochs + epoch + 1),
         'cost =', '{:.9f}'.format(avg_cost))
    print('Accuracy:', accuracy)
train_writer.add_summary(summary)
print('Learning Finished!')

# Test model and check accuracy
X_t,Y_t = test_batch()
print('Accuracy:', m1.get_accuracy(X_t, Y_t))
