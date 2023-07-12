import face_recognition #compares faces
import cv2 #takes input from webcam and gives to fr
import numpy as np #for numpy array
import csv #handle csv file
import os #to access file csv
from datetime import datetime #get exact date and time

video_capture = cv2.VideoCapture(0) #0 means input from default webcam

jobs_image = face_recognition.load_image_file("images\steve.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

wills_image = face_recognition.load_image_file("images\will.jpg")
wills_encoding = face_recognition.face_encodings(wills_image)[0]

einstein_image = face_recognition.load_image_file("images\Einstein.jpg")
einstein_encoding = face_recognition.face_encodings(einstein_image)[0]


known_face_encodings = [
    jobs_encoding, wills_encoding, einstein_encoding
]

known_face_names = [
    "jobs", "will", "einstein"
]

members  = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s= True

now = datetime.now()
current_date = now.strftime("%d/%m/%Y")


f = open(current_date+".csv", "w+", newline="") #f is the name of the csv, #write+
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()#first value which is signal is not needed. we only need frames
    small_frame =  cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:,:,::-1])
    if s: 
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance) # to get the best fit of the data being received using numpy
            if matches[best_match_index]:
                name=known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in members:
                    members.remove(name)
                    print(members)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])

    cv2.imshow("attendance system", frame) #show to user from opencv
    if cv2.waitKey(1) & 0xFF == ord('q'): #executed when we click q button
        break


video_capture.release()
cv2.destroyAllWindows()
f.close()




