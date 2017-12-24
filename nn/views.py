import base64

from django.shortcuts import render, redirect
from django.views import generic
import numpy as np
import json
import itertools
import random
from io import BytesIO
from .util import save_value, save_list
from .multi_layer_net_extend import MultiLayerNetExtend
from .trainer import Trainer


app_name = 'nn'
app_index = app_name + ':index'
tmp_params = 'tmp/' + app_name +'/test_params.pkl'

class Home(generic.TemplateView):
    template_name = app_name + '/index.html'

class Accuracies:
    def __init__(self, trains, tests):
        self.list = []
        for i in range(len(trains)):
            self.list.append(Accuracy(trains[i], tests[i]))

class Accuracy:
    def __init__(self, train, test):
        self.train = train
        self.test = test

class Results:
    def __init__(self, params, labels):
        self.list = []
        for i in range(params.shape[0]):
            self.list.append(Result(params[i], labels[i]))

class Result:
    def __init__(self, param, label):
        self.param = param
        self.label = label
        self.str_param = np.array2string(param, separator=', ')

class Train:
    def __init__(self):
        self.current_hsl = []
        self.current_wdl = 0
        self.current_op_lr = 0
        self.result_acc = 0
        self.result_hsl = []
        self.result_wdl = 0
        self.result_op_lr = 0

    def update(self, acc, hsl, wdl, op_lr):
        self.current_hsl = hsl
        self.current_wdl = wdl
        self.current_op_lr = op_lr
        self.save_current()
        if self.result_acc > acc:
            return False
        self.result_acc = acc
        self.result_hsl = hsl
        self.result_wdl = wdl
        self.result_op_lr = op_lr
        self.save_result()
        return True

    def save_current(self):
        save_list('tmp/nn/train/current_hsl.txt', self.current_hsl)
        save_value('tmp/nn/train/current_wdl.txt', self.current_wdl)
        save_value('tmp/nn/train/current_op_lr.txt', self.current_op_lr)

    def save_result(self):
        save_value('tmp/nn/train/result_acc.txt', self.result_acc)
        save_list('tmp/nn/train/result_hsl.txt', self.result_hsl)
        save_value('tmp/nn/train/result_wdl.txt', self.result_wdl)
        save_value('tmp/nn/train/result_op_lr.txt', self.result_op_lr)


def hidden_size_list(layer_num, max_node):
    node_list = []
    seq = range(1, max_node + 1)
    for i in seq:
        node_list.append([i])

    if layer_num <= 1:
        return node_list
    for i in range(2, layer_num + 1):
        for tpl in itertools.combinations_with_replacement(seq, i):
            node_list.append(list(tpl))

    return node_list


def train(request):
    if request.method != 'POST':
        return redirect(app_index)

    learn_file = request.FILES.get("train_file")
    if learn_file == None:
        return redirect(app_index)

    learn_dict = json.load(learn_file)
    keys = learn_dict["keys"]
    data = np.array(learn_dict["data"])
    random.shuffle(data)
    params, result = np.hsplit(data, [data.shape[1] - 1])
    x_test, x_train = np.vsplit(params, [data.shape[0] // 3])
    t_test, t_train = np.vsplit(result, [data.shape[0] // 3])
    input_size = params.shape[1] * 1

    train_data = Train()
    for hsl in hidden_size_list(1, 1):
        for wdl in np.arange(1, 2, 1):
            for op_lr in np.arange(0.001, 0.002, 0.001):
                network = MultiLayerNetExtend(input_size=input_size, hidden_size_list=hsl,
                                            output_size=1, activation='sigmoid', weight_init_std='sigmoid',
                                            weight_decay_lambda=wdl, use_dropout=False)
                trainer = Trainer(network, x_train, t_train, x_test, t_test,
                                epochs=10, mini_batch_size=2,
                                optimizer='adam', optimizer_param={'lr': op_lr}, verbose=True)
                trainer.train()
                train_data.update(trainer.result_acc, hsl, wdl, op_lr)

    accuracies = []
    context = {
        "accuracies": accuracies
    }

    return render(request, app_name + '/train.html', context)


def learn(request):
    if request.method != 'POST':
        return redirect(app_index)

    learn_file = request.FILES.get("learn_file")
    if learn_file == None:
        return redirect(app_index)

    learn_dict = json.load(learn_file)
    keys = learn_dict["keys"]
    data = np.array(learn_dict["data"])
    random.shuffle(data)
    params, result = np.hsplit(data, [data.shape[1] - 1])
    x_test, x_train = np.vsplit(params, [data.shape[0] // 5])
    t_test, t_train = np.vsplit(result, [data.shape[0] // 5])

    input_size = params.shape[1] * 1
    network = MultiLayerNetExtend(input_size=input_size, hidden_size_list=[15],
                                output_size=65, weight_decay_lambda=0.000001, use_dropout = True, dropout_ration = 0.1)
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                    epochs=500000, mini_batch_size=10,
                    optimizer='adam', optimizer_param={'lr': 0.000001}, verbose=True)
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

    input_size = params.shape[1] * 1
    network = MultiLayerNetExtend(input_size=input_size, hidden_size_list=[10, 10, 10],
                                output_size=65, use_dropout=False)
    network.load_params('tmp/nn/test_params.pkl')

    labels = network.predict(params).argmax(axis=1).reshape(-1, 1)
    results = Results(params, labels)

    context = {
        'results': results
    }

    return render(request, app_name + '/result.html', context)
