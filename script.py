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
font = cv2.FONT_HERSHEY_SIMPLEX

# Basically return a number based on the direction an eye is looking at
def getGazeRatio(eye_points, face_landmarks):
    tmp = []
    for point in eye_points:
        tmp.append((face_landmarks[point][0], face_landmarks[point][1]))
    region = np.array(tmp)
    
    # Creating a mask to isolate the eye and remove background
    height, width, _ = frame.shape 
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [leftEye], True, 255, 2)
    cv2.fillPoly(mask, [leftEye], 255)
    left_eye = cv2.bitwise_and(gray, gray, mask=mask)

    # Isolate a frame for the eyes only
    min_x = np.min(leftEye[:, 0])
    max_x = np.max(leftEye[:, 0])
    min_y = np.min(leftEye[:, 1])
    max_y = np.max(leftEye[:, 1])
    # eye = frame[min_y: max_y, min_x: max_x]
    gray_eye = left_eye[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0:height, 0:int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0:height, int(width/2):width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    # threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    # eye = cv2.resize(eye, None, fx=5, fy=5)
    # cv2.putText(frame, str(right_side_white), (50, 400), font, 2, (0, 0, 255), 3)
    # cv2.putText(frame, str(left_side_white), (50, 200), font, 2, (0, 0, 255), 3)
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white/right_side_white
    return gaze_ratio

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


        
        
        # Display isolated eyes
        # cv2.imshow("Left Eye", eye)
        # cv2.imshow("BW Left Eye", threshold_eye)


        # Gaze detection
        left_gaze_ratio = getGazeRatio(list(range(36,42)), face_landmarks)
        right_gaze_ratio = getGazeRatio(list(range(42,48)), face_landmarks)
        gaze_ratio = (left_gaze_ratio+right_gaze_ratio)/2
        # cv2.putText(frame, str(gaze_ratio), (50, 400), font, 2, (0, 0, 255), 3)
        if gaze_ratio <= 0.4:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
        elif 0.4 < gaze_ratio < 1.8:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)

        # Outline the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
        # 

        # Count how many frames a person's eyes have been closed for
        if EAR < EAR_THRESH:
            COUNTER += 1
        
            if COUNTER >= BLINK_THRESH:
                cv2.putText(frame, "PAY ATTENTION!", (10, 30), font, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
        

        # for n in range(36, 48):
        #     x = face_landmarks.part(n).x
        #     y = face_landmarks.part(n).y
        #     cv2.circle(frame, (x,y), 1, (0, 255, 255), 1)
    
    cv2.imshow("Face", frame)

    key = cv2.waitKey(1)
    if key==27:
        break
    
cap.release()
cv2.destroyAllWindows()