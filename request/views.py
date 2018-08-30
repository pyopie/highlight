from django.shortcuts import render

# Create your views here.

def index(request):
    for f in request.FILES.getlist('files_video'):
        print(f.name)
    return render(request, 'request/layout_request.html',)