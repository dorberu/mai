# coding: utf-8

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import math
from typing import *
from .optimizer import *
from .multi_layer_net_extend import *

class Trainer:
    """ニューラルネットの訓練を行うクラス

    Parameters
    ----------
    network : 拡張版の全結合による多層ニューラルネットワーククラス(MultiLayerNetExtend)
    x_train : 訓練用パラメータ(e.g. array([[1,2,3],[2,3,4],[3,4,5]]))
    t_train : 訓練用教師データ(e.g. array([[5],[10],[17]]))
    x_test : テスト用パラメータ(e.g. array([[1,1,1],[5,10,15]]))
    t_test : テスト用正解データ(e.g. array([[3],[65]]))
    epochs : 1つの訓練データを反復学習させる回数(e.g. 20)
    mini_batch_size : 1つのバッチに含まれる訓練データ数(e.g. 100)
    optimizer : 勾配法の名前(e.g. sgd, momentum)
    optimizer_param : 勾配法のパラメータ(e.g. {'lr':0.01, 'momentum':0.9})
    evaluate_sample_num_per_epoch : 1epoch毎の評価に使用するサンプルデータ数(e.g. 5)
    verbos : 詳細ログの出力有無(e.g. True, False)
    """
    def __init__(self, network: MultiLayerNetExtend,
                 x_train: np.ndarray, t_train: np.ndarray,
                 x_test: np.ndarray, t_test: np.ndarray,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network  # type: MultiLayerNetExtend
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param) # type: Optimizer
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_acc_list = []
        self.test_acc_list = []
        self.result_acc = 0

    def evaluate(self) -> int:
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1
        return self.current_iter

    def train_step(self) -> int:
        # [0 〜 train_size-1]のいずれかの値を持つ整数をbatch_size個生成
        batch_mask = np.random.choice(self.train_size, self.batch_size) # type: np.ndarray
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        loss = self.network.loss(x_batch, t_batch)

        if self.verbose: print("train loss:" + str(loss))
        if math.isnan(loss): return -1
        return self.evaluate()

    def train(self):
        for i in range(self.max_iter):
            if self.train_step() < 0: break

        self.result_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(self.result_acc))
