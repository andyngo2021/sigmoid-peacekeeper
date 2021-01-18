# Collection of useful functions I needed to use in both main.py and calibrate.py

from scipy.spatial import distance as dist
from math import sqrt

def getEyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    EAR = (A+B)/(2*C)
    return EAR

# Get distance between points a (x1, y1) and b (x2, y2)
def getDist(a, b):
    return sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)