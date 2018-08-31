from django.shortcuts import render
import cv2
from .models import *
from .r_functions import *
import face_recognition
import os
from  highlight.settings import *
from io import BytesIO
import time
import zipfile
from django.http import HttpResponse
# Create your views here.
M_path = os.path.join(BASE_DIR, 'media')
file_path_base = M_path + str(os.sep)

def index(request):
    if request.method == "POST":
        if ('file' in request.POST):
            high =None
            for f in request.FILES.getlist('files_video'):
                print(f.name)
                high = Highlight(file_in = f, file_out = f, fname = f.name)
                high.save()
                for i in request.FILES.getlist('files_face'):
                    f_img = Face_img(img = i, fname = i.name)
                    f_img.save()
                    high.f_img.add(f_img)
                    high.save()
                # Open the input movie file
                print("file url",high.file_in.url)
                print("BASE_DIR url", BASE_DIR)
                input_movie = cv2.VideoCapture(BASE_DIR + high.file_in.url)
                length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
                w = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(w, h)
                # Create an output movie file (make sure resolution/frame rate matches input video!)
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')

                output_movie = cv2.VideoWriter(BASE_DIR + high.file_out.url, fourcc, 29.97, (w, h))
                f_img_list = high.f_img.all().order_by('pk')
                known_faces = []
                known_faces_name = []
                for face_img in f_img_list:
                    image = face_recognition.load_image_file(BASE_DIR + face_img.img.url)
                    if (len(face_recognition.face_encodings(image))>0):
                        face_encoding = face_recognition.face_encodings(image)[0]
                    else:
                        return render(request, 'request/message.html', {'msg': "잘못된 얼굴사진!"})
                    known_faces.append(face_encoding)
                    known_faces_name.append(face_img.fname)

                print(len(f_img_list))
                # Initialize some variables
                face_locations = []
                face_encodings = []
                face_names = []
                frame_number = 0

                while True:
                    # Grab a single frame of video
                    ret, frame = input_movie.read()
                    frame_number += 1

                    # Quit when the input video file ends
                    if not ret:
                        break

                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_frame = frame[:, :, ::-1]

                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

                        # If you had more than 2 faces, you could make this logic a lot prettier
                        # but I kept it simple for the demo
                        name = None
                        i=0
                        print("len match = ", len(match))
                        print("len match = ", match)
                        if len(match)>0:
                            for m in match:
                               print("i = ",i)
                               print("m = ",m)
                               if m:
                                   name = known_faces_name[i]
                               i = i+1
                            face_names.append(name)
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        if not name:
                            continue

                        # Draw a box around the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                    # Write the resulting image to the output video file
                    print("Writing frame {} / {}".format(frame_number, length))
                    output_movie.write(frame)
                high.status = 1
                high.save()
            return render(request, 'request/layout_request.html',{'msg': "작업 완료"} )
        elif ('download' in request.POST):
            print("download")
            high_list = Highlight.objects.filter(status=1)
            print("len = ", len(high_list))
            in_memory = BytesIO()
            now = time.localtime()
            fname = "%04d%02d%02d-%02d%02d%02d.zip" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
            zfile = zipfile.ZipFile(in_memory, 'w', zipfile.ZIP_DEFLATED)
            for high in high_list:
                print("high.file_out.name = ", high.file_out.name)
                print("high.fname = ", high.fname)
                zfile.write(file_path_base + high.file_out.name, str(high.pk)+"_out_"+high.fname )
                high.delete()
            zfile.close()
            response = HttpResponse()
            response['content_type'] = 'application/zip'
            response['Content-Disposition'] = 'attachment;filename=' + fname

            in_memory.seek(0)
            response.write(in_memory.read())
            return response

    else:
        return render(request, 'request/layout_request.html',)