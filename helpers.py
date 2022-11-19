from imutils import face_utils
import dlib
import cv2
import numpy as np


def eye_on_mask(mask, side,shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask
def get_frontal_face(rects):
    area=0
    rect_face=0
    for rect in rects:
        x,y,w,h=face_utils.rect_to_bb(rect)
        area_= (x+w)*(y+h)
        if area_> area:
            rect_face=rect
    return rect_face
def brightness(image,gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(image, lookUpTable)

class eyeDetector:
    def __init__(self):
        self.nombre = "Pablo"