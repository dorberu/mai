from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    url(r'^polls/', include('polls.urls')),
    url(r'^keijiban/', include('keijiban.urls')),
    url(r'^cnn/', include('cnn.urls')),
    url(r'^nn/', include('nn.urls')),
    url(r'^admin/', admin.site.urls),
]
