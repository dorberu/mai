from django.conf.urls import url
from keijiban.views import kakikomi
from . import views

urlpatterns = [
    url(r'^$', kakikomi),
]
