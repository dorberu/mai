from django import forms

class KakikomiForm(forms.Form):
     message1 = forms.CharField(widget=forms.Textarea) 
     message2 = forms.CharField(widget=forms.Textarea) 
