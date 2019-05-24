from flask import Flask
app= Flask(__name__)

import cv2
import numpy as np
import time
import dlib
import csv
import datetime

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

#Initializing Frontal Face Detector
face_detector=dlib.get_frontal_face_detector()
#Initializing Face Shape-Landmark Predictor
shape_predictor=dlib.shape_predictor("shape_predictor.dat")
#Initialzing Face Recognizer model
face_recognizer=dlib.face_recognition_model_v1("dlib_face_recognition.dat")
#Initialzing DLIB image window Windows
win = dlib.image_window()

#Training Data Variables 
face_names=[]
face_vectors=[]

#Storing Euclid Distances between Webcam Face and Stored Faces
face_euclids_list=[]

@app.route("/")
def index():
    with open("html/index.html",'r') as html:
        resultString=html.read()

    return resultString

@app.route("/new_person")
def new_person():
    with open("html/new_person.html",'r') as html:
        resultString=html.read()

    return resultString

@app.route("/train_faces")
def face_training():
    #Deleting previous entries
    del face_names[:]
    del face_vectors[:]
    
    print("Traning on Known_Faces ....")
    #Loading CSV File
    with open('known_faces_m/faces_list.csv','rb') as csvfile:
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

    
    return "face_names: {}  face_vectors: {}".format(len(face_names),len(face_vectors))

@app.route("/detect_face/<camera>")
def detectFaces(camera):

    if camera == 'front':
        #Initialzing Video Capture Device
        video_capture=cv2.VideoCapture("rtsp://192.168.1.119:554/11")
        
    if camera == 'back':
        #Initialzing Video Capture Device
        video_capture=cv2.VideoCapture("rtsp://192.168.1.120:554/11")
        
    if camera == 'webcam':
        #Initialzing Video Capture Device
        video_capture=cv2.VideoCapture(0)

    #Set to 30FPS
    video_capture.set(5,30)

    #Face Counter
    face_count=0

    #XML Variable Declatation
    detection=ET.Element("detection")
    timestamp=ET.SubElement(detection,"timestamp")
    timestamp.text=str(datetime.datetime.now())
    faces_xml=ET.SubElement(detection,"faces")
    

    #Grab a frame from the video_capture device
    ret,frame=video_capture.read()

    if not ret:
        return "<br><b>The camera has failed to initialise!Please Restart/Reset your camera.</b></br>"
    if ret:
        
        #Launch Camera Window
        #win.clear_overlay()
        #win.set_image(frame)

        #Detecting Faces from frame, image upscaling=2.
        faces=face_detector(frame,2)
                
        #Printing the number of faces detected.
        print("Number of faces detected: {}".format(len(faces)))

     
        #Runs through each face and extract metrics. 
        for i,d in enumerate(faces):
            #Clear Face_Euclids_list
            del face_euclids_list[:]
            #Face Location Info
            face_count=face_count+1
            left=d.left()
            right=d.right()
            top=d.top()
            bottom=d.bottom()
            print("Detection {} Left: {} Top: {} Right: {} Bottom: {}".format(i,left,top,right,bottom))
            #Adding detected Face coordinates to list
            
            #Declaring XML Elements
            face_xml=ET.SubElement(faces_xml,"face")
            face_id_xml=ET.SubElement(face_xml,"face_id")
            left_xml=ET.SubElement(face_xml,"left")
            right_xml=ET.SubElement(face_xml,"right")
            top_xml=ET.SubElement(face_xml,"top")
            bottom_xml=ET.SubElement(face_xml,"bottom")
            #Appending XML Elements with values
            face_id_xml.text=str(i)
            left_xml.text=str(left)
            right_xml.text=str(right)
            top_xml.text=str(top)
            bottom_xml.text=str(bottom)

            
            #Get the landmarks of the face from box 'd'
            landmarks=shape_predictor(frame,d)
                        
                        
            #Load Face Descriptor
            face_descriptor=face_recognizer.compute_face_descriptor(frame,landmarks)
     

            #Finding all Euclidian Distances
            #Calling the face_vectors list to compare with the newly detected face
            for vector in face_vectors:
                
                #Converting Vectors to Numpy Arrays
                #Converting Stored Face Vector
                known_face=np.array(vector)
                #Converting New Face Vector
                new_face=np.array(face_descriptor)

                #Calculating Euclid's Distance
                euc_dist=np.linalg.norm(known_face-new_face)
                face_euclids_list.append(euc_dist)
                print("Euclid Distance: {}".format(euc_dist))
                
            #Finding the index of the minimum value in face_euclids_list
            val, idx = min((val, idx) for (idx, val) in enumerate(face_euclids_list))

            #If minimum value>0.6, Intruder allert
            if val>0.6:
                print("Unknown Face")
                #Declaring XML Elements
                unknownface=ET.SubElement(face_xml,"unknownface")
                #Appending XML Elements with values
                unknownface.text="Unknown Face"

                
 
            #Checking if known_face, threshold=0.6
            if val<0.6:
                print(idx)
                person_name=face_names[idx]
                prc_conf= round((val/0.6)*100,2)
                print("Person Name: {} Confidence: {}%".format(person_name,prc_conf))
                #Declaring XML Elements
                knownface=ET.SubElement(face_xml,"knownface")
                face_name=ET.SubElement(knownface,"facename")
                confidence=ET.SubElement(knownface,"confidence")
                #Appending XML Elements with Values
                face_name.text=person_name
                confidence.text=str(prc_conf)
                
            
        #Release Video Capture Device
        video_capture.release()

        #Prettifying XML String
        xml_string=ET.tostring(detection, encoding='utf8', method='xml')
        reparsed = minidom.parseString(xml_string)
        return reparsed.toprettyxml(indent="\t")
 
    
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')


    
    


