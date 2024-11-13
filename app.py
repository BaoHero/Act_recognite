import mediapipe as mp 
import cv2
import pandas as pd 

#Read image
cap = cv2.VideoCapture(0)

#Create mediapipe lib
mpPose = mp.solutions.pose 
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []

label = "HeadSwing"
no_of_frames = 100
i = 0

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

while len(lm_list) <= no_of_frames :
    ret, frame = cap.read()
    if ret:
        #Realize pose
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            #Save action bone line
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            #Draw action bone line
            frame = draw_landmark_on_image(mpDraw, results, frame)
        cv2.imshow("image" , frame)
        if cv2.waitKey(1) == ord('q'):
            break

#Write to csv
df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")

cap.release()
cv2.destroyAllWindows()


