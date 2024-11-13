import mediapipe as mp 
import cv2
import pandas as pd 
import numpy as np
import threading
import tensorflow as tf 

label = "Your action ?"

#Read image
cap = cv2.VideoCapture(0)

#Create mediapipe lib
mpPose = mp.solutions.pose 
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

#Load model
model = tf.keras.models.load_model("Action_Human_Model.h5")

n_time_steps = 10
lm_list = []


def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility) 
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    #Draw a straige line
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    
    #Draw a point on line
    for id , lm in enumerate(results.pose_landmarks.landmark):
        h,w,c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx,cy), 10, (0, 0, 255), cv2.FILLED)
    return img

def draw_class_on_image(label, img): #Draw text on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 0, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    print(results)
    if results[0][0] > 0.5:
        label = "Your body is moving !"
    else:
        label = "You swing your hand !"
    return label

i = 0
warmup_frames = 60

while True :
    ret, frame = cap.read()
    #Realize pose
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    i = i+1
    if i > warmup_frames:
        print("Start detect...")

        if results.pose_landmarks:
            #Save action bone line
            lm = make_landmark_timestep(results)
            lm_list.append(lm)

            if len(lm_list) == n_time_steps:
                #Get in to model
                thread_01 = threading.Thread(target=detect, args=(model, lm_list,))
                thread_01.start()
                lm_list = []

                #Print result


            #Draw action bone line
            frame = draw_landmark_on_image(mpDraw, results, frame)
        frame = draw_class_on_image(label , frame)

        cv2.imshow("image" , frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


