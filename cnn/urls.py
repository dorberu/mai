from django.conf.urls import url
from . import views
from cnn.views import Home, upload, learn

app_name = 'cnn'
urlpatterns = [
    url(r'^$', Home.as_view(), name='index'),
    url(r'^upload/$', upload, name='upload'),
    url(r'^learn/$', learn, name='learn'),
]
