from django.conf.urls import url
from . import views
from nn.views import Home, upload, learn, train

app_name = 'nn'
urlpatterns = [
    url(r'^$', Home.as_view(), name='index'),
    url(r'^upload/$', upload, name='upload'),
    url(r'^learn/$', learn, name='learn'),
    url(r'^train/$', train, name='train'),
]
