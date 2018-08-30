from django.db import models

# Create your models here.

def upload_to_input(instance, filename):
    return 'file/input/%s' % (filename)

def upload_to_output(instance, filename):
    return 'file/output/%s' % (filename)

def upload_to_img(instance, filename):
    return 'file/img/%s' % (filename)

class Face_img(models.Model):
    img = models.FileField(blank=True, null=True, upload_to=upload_to_img)
    fname = models.CharField(max_length=30, null=True)

class Highlight(models.Model):
    f_img = models.ManyToManyField(Face_img, blank=True, null=True)
    file_in = models.FileField(blank=True, null=True, upload_to=upload_to_input)
    fname = models.CharField(max_length=30, null=True)
    file_out = models.FileField(blank=True, null=True, upload_to=upload_to_output)