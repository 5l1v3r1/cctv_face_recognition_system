import cv2
import numpy as np
import time
import dlib
import csv
import threading

import pyttsx
import dweepy
import datetime
#RTMP Stream Front Camera:  rtmp://192.168.1.119:1935/flash/11:YWRtaW46YWRtaW4=
#RTMP Stream Back Camera:  rtmp://192.168.1.120:1935/flash/11:YWRtaW46YWRtaW4=

#Face Counter
face_count=0


#Initializing Frontal Face Detector
face_detector=dlib.get_frontal_face_detector()
#Initializing Face Shape-Landmark Predictor
shape_predictor=dlib.shape_predictor("shape_predictor.dat")
#Initialzing Face Recognizer model
face_recognizer=dlib.face_recognition_model_v1("dlib_face_recognition.dat")
#Initializing Face Tracker
face_tracker=dlib.correlation_tracker()

#Creating an Open CV Font
cv_font =cv2.FONT_HERSHEY_SIMPLEX



#Initialzing DLIB image window Windows
win = dlib.image_window()

#Training Data Variables 
face_names=[]
face_vectors=[]

#Storing Euclid Distances between Webcam Face and Stored Faces
face_euclids_list=[]

def doTraining(filepath):
    print("Traning on Known_Faces ....")
    with open(filepath,'rb') as csvfile:
        readCSV=csv.reader(csvfile,delimiter=',')
        for row in readCSV:
            print("Training on image: {} @ {}".format(row[0],row[1]))
            #Load Image From File
            image=cv2.imread(row[1])
            #Convert BGR to RGB
            face_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            #Detect Faces from 'face_image'. Upscaling =1 
            det_faces=face_detector(face_image,1)
            #There should be atleast on Face.
            #Traning images has one face per image.
            if len(det_faces)==0:
                print("Face Image invalid or not found!")
            if len(det_faces)>0:
                for i,d in enumerate(det_faces):
                    #Print Face Details
                    print("Face Found! Left: {} Right: {} Top: {} Bottom: {}".format(d.left(),d.right(),d.top(),d.bottom()))
                    
                    #Detect Landmarks on Face
                    landmarks=shape_predictor(face_image,d)
                    #Compute Face_Descriptor Vector
                    face_descriptor=face_recognizer.compute_face_descriptor(face_image,landmarks)

                    #Append Lists
                    face_names.append(row[0])
                    face_vectors.append(face_descriptor)

    print("Known_Faces training compleate!")
    print ("face_names: {}  face_vectors: {}".format(len(face_names),len(face_vectors)))

def addText(frame,text,x,y):
    cv2.putText(frame,text,(x,y),cv_font,0.5,(0,255,0),1,cv2.LINE_AA)
  

doTraining("known_faces_m/faces_list.csv")


while (True):
    #Image Processing Start Time
    start_time=time.clock()
    
    #Initialzing Video Capture Device
    #RSTP URL: "rtsp://192.168.1.119:554/11"
    video_capture=cv2.VideoCapture(0)
    #Set to 30FPS
    video_capture.set(5,25)
    #Set Video Resolution to 640x480
    video_capture.set(3,640) #Width
    video_capture.set(4,480) #Height
                
    #Grab a frame from the video_capture device
    ret,frame=video_capture.read()
    #Release Camera
    video_capture.release()

    #Getting the current Time
    current_time=str(datetime.datetime.now())
    addText(frame,current_time,10,20)

    
    
    
    if ret:
        #Convert BGR to RGB
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
       

        #Detecting Faces from frame, image upscaling=2.
        faces=face_detector(rgb_frame,2)

        #Launch Camera Window
        win.clear_overlay()
        win.set_image(rgb_frame)

                        
        #Printing the number of faces detected.
        print("Number of faces detected: {}".format(len(faces)))

        #Run through each face and extract metrics. 
        for i,d in enumerate(faces):
            #Clear Face_Euclids_list
            del face_euclids_list[:]
            #Face Detection Coordinates
            face_count=face_count+1
            left=d.left()
            right=d.right()
            top=d.top()
            bottom=d.bottom()
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i,left,top,right,bottom))

            #Tracking a Detected Face
            #face_tracker.start_track(frame,dlib.rectangle(left,right,top,bottom))

            #Get the landmarks of the face from box 'd'
            landmarks=shape_predictor(rgb_frame,d)
            #Adding Overlays
            win.add_overlay(landmarks)       
                        
            face_descriptor=face_recognizer.compute_face_descriptor(rgb_frame,landmarks)
     

            #Finding all Euclidian Distances
            for vector in face_vectors:
                
                #Converting Vectors to Numpy Arrays
                known_face=np.array(vector)
                new_face=np.array(face_descriptor)

                #Calculating Euclid's Distance
                euc_dist=np.linalg.norm(known_face-new_face)
                face_euclids_list.append(euc_dist)
                print("Euclid Distance: {}".format(euc_dist))
                
            #Finding the index of the minimum value in face_euclids_list
            val, idx = min((val, idx) for (idx, val) in enumerate(face_euclids_list))

            #If minimum value>0.6, Intruder allert
            if val>0.6:
                print("Intruder Alert")
                label="Unknown Person"
                addText(rgb_frame,label,left,top)
             

            #Checking if known_face, threshold=0.6
            if val<0.6:
                print(idx)
                person_name=face_names[idx]
                prc_conf=round((val/0.6)*100,2)
                print("Person Name: {} Confidence: {}%".format(person_name,prc_conf))
                label ="Person Name: {} Confidence: {}%".format(person_name,prc_conf)
                addText(rgb_frame,label,left,top)

        #Getting Image Processing End Time
        improc_freq="Frequency: {}Hz".format(round(1/(time.clock()-start_time),4))
        addText(rgb_frame,improc_freq,10,40)
        win.set_image(rgb_frame)

    
                
                
              


  
