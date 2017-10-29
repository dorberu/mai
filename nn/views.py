import base64

from django.shortcuts import render, redirect
from django.views import generic
import numpy as np
import json
import random
from io import BytesIO
from nn.multi_layer_net_extend import MultiLayerNetExtend
from nn.trainer import Trainer


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
