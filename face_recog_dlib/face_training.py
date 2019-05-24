import cv2
import numpy as np
import time
import dlib
import csv
import threading
import pyttsx
#RTMP Stream Front Camera:  rtmp://192.168.1.119:1935/flash/11:YWRtaW46YWRtaW4=
#RTMP Stream Back Camera:  rtmp://192.168.1.120:1935/flash/11:YWRtaW46YWRtaW4=

#Initialzing Video Capture Device
video_capture=cv2.VideoCapture(0)
#Set to 30FPS
video_capture.set(5,30)
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
  

doTraining("known_faces_m/faces_list.csv")


while (True):
    
    
    #Grab a frame from the video_capture device
    ret,frame=video_capture.read()

    #If the video gets stuck.Restart Capture_Device.
    if ret is False:
        print("Video is Stuck. Resetting video...")
        video_capture=cv2.VideoCapture(0)
        print("Video reset!")

    if ret:
        #Convert BGR to RGB
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        #Launch Camera Window
        win.clear_overlay()
        win.set_image(rgb_frame)

        #Detecting Faces from frame, image upscaling=2.
        faces=face_detector(rgb_frame,2)

                        
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
            #win.add_overlay(face_tracker.get_position())
            #print("Face Location : {}".format(face_tracker.get_position()))

            #Load Face Descriptor
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
                
             

            #Checking if known_face, threshold=0.6
            if val<0.6:
                print(idx)
                person_name=face_names[idx]
                prc_conf=(val/0.6)*100
                print("Person Name: {} Confidence: {}%".format(person_name,prc_conf))
                
              

video_capture.release()
  
