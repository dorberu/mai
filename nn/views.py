import base64

from typing import List, Dict
from django.shortcuts import render, redirect
from django.views import generic
import numpy as np
import json
import itertools
import random
from io import BytesIO
from .util import save_value
from .multi_layer_net_extend import MultiLayerNetExtend
from .trainer import Trainer


app_name = 'nn'
app_index = app_name + ':index'
tmp_params = 'tmp/' + app_name +'/test_params.pkl'

class Home(generic.TemplateView):
    template_name = app_name + '/index.html'

class Accuracies:
    def __init__(self, trains, tests):
        self.list = [Accuracy(trains[i], tests[i]) for i in range(len(trains))]

class Accuracy:
    def __init__(self, train, test):
        self.train = train
        self.test = test

class Results:
    def __init__(self, params, labels):
        self.list = [Result(params[i], labels[i]) for i in range(params.shape[0])]

class Result:
    def __init__(self, param, label):
        self.param = param
        self.label = label
        self.str_param = np.array2string(param, separator=', ')

class TrainLog:

    def __init__(self, acc: float, hsl: List[int], wdl: float, op_lr: float):
        self.acc = acc
        self.hsl = hsl
        self.wdl = wdl
        self.op_lr = op_lr

    def save_current(self):
        output = [ 'acc:' + str(self.acc) ]
        output += [ 'hsl:[' + ','.join(map(str, self.hsl)) + ']' ]
        output += [ 'wdl:' + str(self.wdl) ]
        output += [ 'op_lr:' + str(self.op_lr) ]
        save_value('tmp/nn/train/current.txt', ','.join(output))


def hidden_size_list(max_layer_num: int, max_node: int) -> List[List[int]]:
    """隠れ層の全パターンリストを作成

    Parameters
    ----------
    max_layer_num : 隠れ層の最大数
    max_node : 各層のノードの最大数

    Return
    ----------
    node_list : 隠れ層の全パターンリスト

    Example
    ----------
    max_layer_num = 3, max_node = 2
    => [[1],[2],[1,1],[1,2],[2,2],[1,1,1],[1,1,2],[1,2,2],[2,2,2]]
    """
    node_list = []  # type: List[List[int]]
    seq = range(1, max_node + 1)
    for i in range(1, max_layer_num + 1):
        node_list += [list(tpl) for tpl in itertools.combinations_with_replacement(seq, i)]
    return node_list


def train(request):
    if request.method != 'POST': return redirect(app_index)
    train_file = request.FILES.get('train_file')
    if train_file is None: return redirect(app_index)

    learn_dict = json.load(train_file)
    keys = learn_dict['keys']
    data = np.array(learn_dict['data'])
    np.random.shuffle(data)
    params, result = np.hsplit(data, [data.shape[1] - 1])       # パラメータと正解データに分解
    x_test, x_train = np.vsplit(params, [data.shape[0] // 3])   # テスト用パラメータと訓練用パラメータ
    t_test, t_train = np.vsplit(result, [data.shape[0] // 3])   # テスト用正解データと訓練用教師データ
    input_size = params.shape[1]    # パラメータの個数

    train_log_list = []
    for hsl in hidden_size_list(3, 5):
        for wdl in np.arange(0, 0.5, 0.001):
            for op_lr in np.arange(0.001, 0.5, 0.001):
                network = MultiLayerNetExtend(input_size=input_size, hidden_size_list=hsl,
                                            output_size=1, activation='relu', weight_init_std='relu',
                                            weight_decay_lambda=wdl, use_dropout=False)
                trainer = Trainer(network, x_train, t_train, x_test, t_test,
                                epochs=200, mini_batch_size=10,
                                optimizer='adam', optimizer_param={'lr': op_lr}, verbose=True)
                trainer.train()

                train_log = TrainLog(trainer.result_acc, hsl, wdl, op_lr)
                train_log.save_current()
                train_log_list += [ train_log ]

    context = {
        "train_log_list": train_log_list
    }

    return render(request, app_name + '/train.html', context)


def learn(request):
    if request.method != 'POST': return redirect(app_index)
    learn_file = request.FILES.get("learn_file")
    if learn_file is None: return redirect(app_index)

    learn_dict = json.load(learn_file)
    keys = learn_dict["keys"]
    data = np.array(learn_dict["data"])
    np.random.shuffle(data)
    params, result = np.hsplit(data, [data.shape[1] - 1])
    x_test, x_train = np.vsplit(params, [data.shape[0] // 3])
    t_test, t_train = np.vsplit(result, [data.shape[0] // 3])
    input_size = params.shape[1]

    network = MultiLayerNetExtend(input_size=input_size, hidden_size_list=[5],
                                output_size=1, activation='relu', weight_init_std='relu',
                                weight_decay_lambda=0, use_dropout=False)
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                    epochs=2000, mini_batch_size=10, optimizer='adam', verbose=True)
    trainer.train()
    network.save_params("tmp/nn/test_params.pkl")

    accuracies = Accuracies(trainer.train_acc_list, trainer.test_acc_list)

    context = {
        "accuracies": accuracies
    }

    return render(request, app_name + '/learn.html', context)
    

def upload(request):
    if request.method != 'POST':
        return redirect(app_index)

    check_file = request.FILES.get("check_file")
    if check_file == None:
        return redirect(app_index)

    learn_dict = json.load(check_file)
    params = np.array(learn_dict["data"])
    if params.shape[1] != 3:
        return redirect(app_index)

    input_size = params.shape[1]
    network = MultiLayerNetExtend(input_size=input_size, hidden_size_list=[5],
                                output_size=1, activation='relu', weight_init_std='relu',
                                weight_decay_lambda=0, use_dropout=False)
    network.load_params('tmp/nn/test_params.pkl')

    labels = network.predict(params).round().astype(int)
    results = Results(params, labels)

    context = {
        'results': results
    }

    return render(request, app_name + '/result.html', context)
