import base64

from django.shortcuts import render, redirect
from django.views import generic
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
 
from cnn.simple_convnet import SimpleConvNet
 
network = SimpleConvNet(
    input_dim=(1, 28, 28), hidden_size=100, output_size=10)
network.load_params('cnn/params.pkl')
 
 
class Home(generic.TemplateView):
    template_name = 'cnn/index.html'
 
 
def upload(request):
    files = request.FILES.getlist("files[]")
    if request.method == 'POST' and files:
        array_list = []
        for file in files:
            img = Image.open(file)
            img = ImageOps.grayscale(img.resize((28, 28)))
            array = np.asarray(img)
            array_list.append(array)
 
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
    else:
        return redirect('cnn/index.html')
