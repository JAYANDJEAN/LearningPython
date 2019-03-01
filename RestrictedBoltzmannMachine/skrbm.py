# coding:utf-8
import time
import numpy as np
import pandas as pd
import datetime
from sklearn.neural_network import BernoulliRBM


class BRBM(object):
    def __init__(self, _n_visible, _n_hiddens=2, _learning_rate=0.1, _batch_size=10, _n_iter=300):
        self.n_hiddens = _n_hiddens
        self.learning_rate = _learning_rate
        self.batch_size = _batch_size
        self.n_iter = _n_iter

        self.components_ = np.asarray(np.random.normal(0, 0.01, (_n_hiddens, _n_visible)), order='fortran')
        self.intercept_hidden_ = np.zeros(_n_hiddens, )
        self.intercept_visible_ = np.zeros(_n_visible, )
        self.h_samples_ = np.zeros((_batch_size, _n_hiddens))

    def transform(self, x):
        h = self._mean_hiddens(x)
        return h

    def reconstruct(self, x):
        h = self._mean_hiddens(x)
        v = self._mean_visibles(h)
        return v

    def gibbs(self, v):
        h_ = self._sample_hiddens(v)
        v_ = self._sample_visibles(h_)
        return v_

    def fit(self, x):
        n_samples = x.shape[0]
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(self._slices(n_batches * self.batch_size, n_batches, n_samples))
        for iteration in xrange(self.n_iter):
            for batch_slice in batch_slices:
                self._fit_pcd(x[batch_slice])
        return self

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _mean_hiddens(self, v):
        p = np.dot(v, self.components_.T)
        p += self.intercept_hidden_
        return self._sigmoid(p)

    def _mean_visibles(self, h):
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        return self._sigmoid(p)

    def _sample_hiddens(self, v):
        p = self._mean_hiddens(v)
        return np.random.random_sample(size=p.shape) < p

    def _sample_visibles(self, h):
        p = self._mean_visibles(h)
        return np.random.random_sample(size=p.shape) < p

    def _free_energy(self, v):
        return (- np.dot(v, self.intercept_visible_)
                - np.logaddexp(0, np.dot(v, self.components_.T)
                               + self.intercept_hidden_).sum(axis=1))

    def _fit_cd(self, v_pos):

        # h_pos=h1;h_neg=h2;
        # v_pos=v1,也就是原始数据;v_neg=v2,也就是重构以后的数据
        # v_pos=K*N;K是单次迭代的样本个数,N是可见单元的个数

        # h_pos=K*M;M是隐藏单元的个数
        # v_neg=K*N

        '''
        h_pos = self._mean_hiddens(v_pos)

        v_neg = self._sample_visibles(self._sample_hiddens(v_pos))
        h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = np.dot(v_pos.T, h_pos).T
        update -= np.dot(h_neg.T, v_neg)

        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr * (v_pos.sum(axis=0) - v_neg.sum(axis=0))
        '''

        v1 = v_pos
        h1 = self._mean_hiddens(v1)
        v2 = self._sample_visibles(self._sample_hiddens(v1))
        h2 = self._mean_hiddens(v2)

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = np.dot(v1.T, h1).T
        update -= np.dot(h2.T, v2)

        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h1.sum(axis=0) - h2.sum(axis=0))
        self.intercept_visible_ += lr * (v1.sum(axis=0) - v2.sum(axis=0))

    def _fit_pcd(self, v_pos):
        """Inner fit for one mini-batch.

        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).

        Parameters
        ----------
        v_pos : array-like, shape (n_samples, n_features)
            The data to use for training.
        """
        h_pos = self._mean_hiddens(v_pos)
        v_neg = self._sample_visibles(self.h_samples_)
        h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = np.dot(v_pos.T, h_pos).T
        update -= np.dot(h_neg.T, v_neg)

        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr * (v_pos.sum(axis=0) - v_neg.sum(axis=0))

        # 更新h_samples_
        # self.h_samples_ = np.random.random_sample(size=h_neg.shape) < h_neg
        h_neg[np.random.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)

    def _slices(self, n, n_packs, n_samples=None):

        start = 0
        if n_packs < 1:
            raise ValueError("gen_even_slices got n_packs=%s, must be >=1" % n_packs)
        for pack_num in range(n_packs):
            this_n = n // n_packs
            if pack_num < n % n_packs:
                this_n += 1
            if this_n > 0:
                end = start + this_n
                if n_samples is not None:
                    end = min(n_samples, end)
                yield slice(start, end, None)
                start = end


class Evaluate(object):
    def __init__(self, _train_data_path, _test_data_path, _n_hiddens, _level,
                 _learning_rate=0.1, _batch_size=10, _n_iter=300):
        self.train_data_path = _train_data_path
        self.test_data_path = _test_data_path
        self.n_hiddens = _n_hiddens
        self.level = _level
        self.learning_rate = _learning_rate
        self.batch_size = _batch_size
        self.n_iter = _n_iter

    def get_data_with_ratings(self):
        # 处理train数据:
        # 将评分数据归一化;
        train_data = pd.read_csv(self.train_data_path, index_col=(0, 1), header=None)
        train_data_series = pd.Series(data=(train_data.values.reshape(1, -1) / float(5))[0], index=train_data.index)
        train_data_series.index.names = ("user", "item")
        # 处理test数据:
        # 同样将评分数据归一化,并取评分大于level的数据作为用户的喜欢数据
        test_data = pd.read_csv(self.test_data_path, index_col=(0, 1), header=None)
        test_data_series = pd.Series(data=(test_data.values.reshape(1, -1) / float(5))[0], index=test_data.index)
        test_data_series = test_data_series[test_data_series > self.level]
        test_data_series.index.names = ("user", "item")
        return train_data_series, test_data_series

    def get_data_no_ratings(self):
        # 处理train数据:
        # 有记录的数据标记为1,无记录的数据标记为0
        train_data = pd.read_csv(self.train_data_path, index_col=(0, 1), header=None)
        train_data_series = pd.Series(data=1.0, index=train_data.index)
        train_data_series.index.names = ("user", "item")
        # 处理test数据:
        # 仅保留test集合下的记录,直接忽略评分
        test_data = pd.read_csv(self.test_data_path, index_col=(0, 1), header=None)
        test_data_series = pd.Series(data=1.0, index=test_data.index)
        test_data_series.index.names = ("user", "item")
        return train_data_series, test_data_series

    def evaluate(self, _if_rating, _if_fill):
        if _if_rating:
            train_data_series, test_data_series = self.get_data_with_ratings()
        else:
            train_data_series, test_data_series = self.get_data_no_ratings()

        train_data_df = train_data_series.unstack(fill_value=self.level / 2) if _if_fill \
            else train_data_series.unstack(fill_value=0.0)
        train_data_array = train_data_df.values
        n_user, n_item = train_data_array.shape

        # 创建模型,并训练,再预测
        rbm = BRBM(n_item, self.n_hiddens, self.learning_rate, self.batch_size, self.n_iter)
        rbm.fit(train_data_array)
        predict_data_array = rbm.reconstruct(train_data_array)

        # 判断预测数据的shape
        assert (train_data_array.shape == predict_data_array.shape)

        # 模型评价:
        # 删除已经看过的
        predict_data_df = pd.DataFrame(data=predict_data_array, index=train_data_df.index,
                                       columns=train_data_df.columns)
        predict_data_series = predict_data_df.stack()
        predict_data_series = predict_data_series[predict_data_series > self.level]
        intersection_test_predict = test_data_series.index.intersection(predict_data_series.index)
        intersection_train_predict = train_data_series.index.intersection(predict_data_series.index)
        num_recommendation = len(predict_data_series) - len(intersection_train_predict)
        num_test = len(test_data_series)
        num_intersection = len(intersection_test_predict)

        return [_if_rating, _if_fill, self.learning_rate, self.batch_size, self.n_iter,
                self.n_hiddens, self.level,
                len(predict_data_series),
                len(intersection_train_predict),
                num_recommendation, num_test, num_intersection,
                float(num_intersection) / float(num_recommendation),
                float(num_intersection) / float(num_test),
                2.0 / (
                    float(num_recommendation) / float(num_intersection)
                    + float(num_test) / float(num_intersection))]


train_data_path = "/Users/Jay/Documents/Data_for_ML/data_movie/1m_data_train.csv"
test_data_path = "/Users/Jay/Documents/Data_for_ML/data_movie/1m_data_test.csv"
time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_file_path = "/Users/Jay/Desktop/rbm-" + time_str + ".txt"

num_hidden_list = [150]
level_list = [.4]
if_rating = True
if_fill = False
learning_rate = 0.1
batch_size = 10
n_iter = 10

title = ["if_rating", "if_fill", "learning_rate", "batch_size", "n_iter",
         "n_hidden", "level", "n_predict", "n_train_predict",
         "n_rec", "n_test", "n_intersection", "precision", "recall", "F"]

with open(save_file_path, 'w') as f:
    f.write(",".join(title) + "\n")

    num = 0
    for num_hidden in num_hidden_list:
        for level in level_list:
            start = time.clock()
            model = Evaluate(train_data_path, test_data_path, num_hidden, level,
                             learning_rate, batch_size, n_iter)
            result = model.evaluate(if_rating, if_fill)
            writeline = ",".join([str(i) for i in result]) + "\n"
            f.write(writeline)
            end = time.clock()
            num += 1
            print("finish:%i;take time:%.2f s" % (num, end - start))
            print(writeline)
