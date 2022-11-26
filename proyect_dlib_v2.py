from imutils import face_utils
import dlib
import cv2
import numpy as np
from helpers import *
import time

############################### THINHS TO DO ################
# 1 -Capture face
# 2- Capture landmasks
# 3- Captura eyes
# 4- Zoom in eyes
# 5- Capture iris 
# 6- Capture iris centet
# 7- Get relative position
# 8- Show position
#Convert landmaks to mask
#http://art-of-electronics.blogspot.com/2021/04/iris-detection-python-opencv.html
#https://towardsdatascience.com/detecting-the-iris-and-changing-its-color-using-opencv-and-dlib-30a6aad122dd
#https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6
#https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6


# the facial landmark predictor
p = "facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
eye_detector=eyeDetector()
while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    rect=get_frontal_face(rects)
    if rect ==0:
        pass
    else:
        rect=get_frontal_face(rects)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #start_time_ns = time.process_time_ns()
        #eye_detector.get_eyes_position(image=image,shape=shape)
        cX_r,cY_r,crop_img_r,cX_l,cY_l,crop_img_l=eye_detector.get_eyes_position(image=image,shape=shape)
        #end_time_ns = time.process_time_ns()
        #timer_ns = end_time_ns - start_time_ns
        #print(timer_ns)
        if cX_r is not None or cX_l is not None :
            cv2.circle(crop_img_r, (cX_r, cY_r), 1, (255, 255, 255), -1)  
            cv2.circle(crop_img_l, (cX_l, cY_l), 1, (255, 255, 255), -1)
        cv2.imshow("ojo_derecho",crop_img_r)
        cv2.imshow("ojo_izquierdo",crop_img_l)
    if cv2.waitKey(1) & 0xFF == ord('q'): # escape when q is pressed
        break

cv2.destroyAllWindows()
cap.release()