import cv2
import dlib
# 
from scipy.spatial import distance as dist
# from imutils.video import VideoStream
from imutils import face_utils
# from threading import Thread
import numpy as np 
import imutils

# Constants
# The minimum eye aspect ratio for an eye to be considered closed
EAR_THRESH = 0.3
# If eyes are closed for at least 48 frames, something is off
BLINK_THRESH = 48
# To count the amount of frames an eye has been closed
COUNTER = 0


# eye aspect ratio becomes 0 when eyes are closed
def getEyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    EAR = (A+B)/(2*C)
    return EAR



cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        face_landmarks = predictor(gray, face)
        # Convert to numpy array
        face_landmarks = face_utils.shape_to_np(face_landmarks)
        
        leftEye = face_landmarks[36:42]
        rightEye = face_landmarks[42:48]
        leftEye_EAR = getEyeAspectRatio(leftEye)
        rightEye_EAR = getEyeAspectRatio(rightEye)
        # average out the EAR of each eye
        EAR = (leftEye_EAR+rightEye_EAR)/2.0

        # Outline the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
        # 

        if EAR < EAR_THRESH:
            COUNTER += 1
        
            if COUNTER >= BLINK_THRESH:
                cv2.putText(frame, "PAY ATTENTION!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
        

        # for n in range(36, 48):
        #     x = face_landmarks.part(n).x
        #     y = face_landmarks.part(n).y
        #     cv2.circle(frame, (x,y), 1, (0, 255, 255), 1)
    
    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key==27:
        break
    
cap.release()
cv2.destroyAllWindows()