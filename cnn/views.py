import base64

from django.shortcuts import render, redirect
from django.views import generic
import numpy as np
import imghdr
import json
from PIL import Image, ImageOps
from io import BytesIO
from cnn.simple_convnet import SimpleConvNet
from cnn.trainer import Trainer


class Home(generic.TemplateView):
    template_name = 'cnn/index.html'


def learn(request):
    if request.method != 'POST':
        return redirect('cnn:index')

    t_file = request.FILES.get("t_file")
    if t_file == None:
        return redirect('cnn:index')

    files = request.FILES.getlist("files[]")
    if len(files) <= 0:
        return redirect('cnn:index')

    json_data = json.load(t_file)
    t_list = []
    x_list = []
    for file in files:
        if imghdr.what(file) != 'png':
            continue
        img = Image.open(file)
        if img.size[0] >= 2048 or img.size[1] >= 2048:
            continue
        img = ImageOps.grayscale(img.resize((28, 28)))

        if json_data.get(file.name) == None:
            continue
        t_list.append(int(json_data[file.name]))

        array = np.asarray(img)
        x_list.append(array)

    if len(t_list) <= 0 or len(t_list) != len(x_list):
        return redirect('cnn:index')

    network = SimpleConvNet(input_dim=(1, 28, 28),
        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
        hidden_size=100, output_size=10, weight_init_std=0.01)

    x_train = np.array(x_list).reshape(len(x_list), 1, 28, 28)
    t_train = np.array(t_list)
    x_test = x_train
    t_test = t_train

    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=1, mini_batch_size=len(t_train),
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=len(t_train))
    trainer.train()

    network.save_params("tmp/cnn/test_params.pkl")

    context = {}
    return render(request, 'cnn/learn.html', context)


def upload(request):
    if request.method != 'POST':
        return redirect('cnn:index')

    files = request.FILES.getlist("files[]")
    if len(files) <= 0:
        return redirect('cnn:index')

    array_list = []
    for file in files:
        if imghdr.what(file) != 'png':
            continue
        img = Image.open(file)
        if img.size[0] >= 2048 or img.size[1] >= 2048:
            continue
        img = ImageOps.grayscale(img.resize((28, 28)))
        array = np.asarray(img)
        array_list.append(array)

    if len(array_list) <= 0:
        return redirect('cnn:index')

    network = SimpleConvNet(input_dim=(1, 28, 28),
        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
        hidden_size=100, output_size=10, weight_init_std=0.01)
    network.load_params('tmp/cnn/test_params.pkl')

    x = np.array(array_list).reshape(len(array_list), 1, 28, 28)
    labels = network.predict(x).argmax(axis=1)
    result = []
    for file, label in zip(files, labels):
        file.seek(0)
        src = base64.b64encode(file.read())
        img = Image.open(file)
        img = ImageOps.grayscale(img.resize((28, 28)))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        out = base64.b64encode(buffer.getvalue())
        result.append((src, out, label))
    context = {
        'result': result,
    }
    return render(request, 'cnn/result.html', context)
