from django.http import HttpResponse
from django.shortcuts import render
from keijiban.forms import KakikomiForm

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def kakikomi(request):
    message1 = ""
    message2 = ""
    if request.method == 'POST':
        f = KakikomiForm(request.POST)
        message1 = request.POST.get("message1")
        message2 = request.POST.get("message2")
    else:
        f = KakikomiForm()
    return render(request, 'keijiban/kakikomiform.html', {'form1': f, 'message1': message1, 'message2': message2})
