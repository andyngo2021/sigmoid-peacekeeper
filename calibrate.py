# calibrate.py calibrates the face and sets the constants in constants.txt
# NOT DONE YET
import cv2
import dlib
from functions import getEyeAspectRatio, getDist
from imutils import face_utils
import numpy as np 
import imutils
from playsound import playsound

# Calibrate for 40 frames
CALIBRATION_FRAMES = 40
NUM_FRAMES = 0
done = False
eyes_calibrated = False
edge_calibrated = False
EARS = []


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')



while not done:
    NUM_FRAMES+=1
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        # put some warning thing
        pass

    for face in faces:
        face_landmarks = predictor(gray, face)
        face_landmarks = face_utils.shape_to_np(face_landmarks)
        leftEye = face_landmarks[36:42]
        rightEye = face_landmarks[42:48]
        LEAR = getEyeAspectRatio(leftEye)
        REAR = getEyeAspectRatio(rightEye)
        # cv2.putText(frame, str(round(LEAR, 2)), (50,400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
        # cv2.putText(frame, str(round(REAR, 2)), (450,400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        AVG_EAR = (LEAR+REAR)/2
        for n in range(0, 68):
            x = face_landmarks[n][0]
            y = face_landmarks[n][1]
            cv2.circle(frame, (x,y), 1, (0, 255, 255), 1)

        if not eyes_calibrated and NUM_FRAMES>50:
            if len(EARS) < CALIBRATION_FRAMES:
                cv2.putText(frame, "CLOSE YOUR EYES", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(frame, "UNTIL YOU HEAR A BEEP", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                EARS.append(AVG_EAR)
            else:
                playsound("beep.mp3")
                eyes_calibrated = True
                BLINK_THRESH = sum(EARS)/len(EARS)
                with open("constants.txt", "w+") as fout:
                    fout.write("True\n")
                    fout.write(str(BLINK_THRESH+0.05))
                break
        elif eyes_calibrated:
            cv2.putText(frame, "YOU MAY NOW CLOSE THE PROGRAM", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow("Face", frame)

    key = cv2.waitKey(1)
    if key==27:
        break
    
cap.release()
cv2.destroyAllWindows()