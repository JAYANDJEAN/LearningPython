# coding:utf-8
import tensorflow as tf
import numpy as np


class OddEven(object):
    def __init__(self, dim_input, hidden_info, optimizer=tf.train.AdamOptimizer()):
        self.x = tf.placeholder(tf.float32, [None, dim_input])
        self.y = tf.placeholder(tf.float32, [None, 2])
        self.hidden = None
        self.dim_input = dim_input

        self.weights = self._initialize_weights(hidden_info)

        for i in hidden_info:
            w_name = 'w' + str(i)
            b_name = 'b' + str(i)
            if i == 1:
                self.hidden = tf.nn.sigmoid(tf.add(tf.matmul(self.x, self.weights[w_name]), self.weights[b_name]))
            else:
                self.hidden = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden, self.weights[w_name]), self.weights[b_name]))
        i = len(hidden_info)
        w_name = 'w' + str(i + 1)
        b_name = 'b' + str(i + 1)
        self.reconstruction = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden, self.weights[w_name]), self.weights[b_name]))
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.y), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self, hidden_info):
        all_weights = dict()
        assert type(hidden_info) == dict
        for i in hidden_info:
            w_name = 'w' + str(i)
            b_name = 'b' + str(i)
            if i == 1:
                all_weights[w_name] = tf.get_variable(w_name, shape=[self.dim_input, hidden_info[i]],
                                                      initializer=tf.contrib.layers.xavier_initializer())
            else:
                all_weights[w_name] = tf.get_variable(w_name, shape=[hidden_info[i - 1], hidden_info[i]],
                                                      initializer=tf.contrib.layers.xavier_initializer())

            all_weights[b_name] = tf.Variable(tf.zeros([hidden_info[i]], dtype=tf.float32))
        i = len(hidden_info)
        w_name = 'w' + str(i + 1)
        b_name = 'b' + str(i + 1)
        all_weights[w_name] = tf.get_variable(w_name, shape=[hidden_info[i], 2],
                                              initializer=tf.contrib.layers.xavier_initializer())
        all_weights[b_name] = tf.Variable(tf.zeros([2], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X, Y):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.y: Y})
        return cost

    def calc_total_cost(self, X, Y):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.y: Y})

    def predict(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})


def get_random_block_from_data(_data_x, _data_y, _batch_size):
    start_index = np.random.randint(0, len(_data_x) - _batch_size)
    return _data_x[start_index:(start_index + _batch_size)], _data_y[start_index:(start_index + _batch_size)]


n_sample_digit = 8
n_sample = 2 ** n_sample_digit

data_x = []
for i in range(n_sample):
    a = [0] * n_sample_digit
    b = str(bin(i))
    for j in range(len(b), 2, -1):
        a[n_sample_digit - len(b) + j - 1] = int(b[j - 1])
    data_x.append(a)
data_x = np.array(data_x).reshape(-1, n_sample_digit)

data_y = []
for i in range(n_sample):
    if i % 2 == 0:
        data_y.append([0, 1])
    else:
        data_y.append([1, 0])
data_y = np.array(data_y)

data_x_train = data_x[0:128]
data_y_train = data_y[0:128]
data_x_test = data_x[128:n_sample]
data_y_test = data_y[128:n_sample]

batch_size = 8

hid_info = {1: 50}
model = OddEven(n_sample_digit, hid_info)
training_epochs = 100

for epoch in range(training_epochs):
    total_batch = int(len(data_x_train) / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = get_random_block_from_data(data_x_train, data_y_train, batch_size)
        cost = model.partial_fit(batch_xs, batch_ys)
        print cost
print data_x_test
print model.predict(data_x_test)
