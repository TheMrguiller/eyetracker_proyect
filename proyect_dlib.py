from imutils import face_utils
import dlib
import cv2
import numpy as np
from helpers import *
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

right = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
left = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cv2.namedWindow('image')
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

        mask_right = np.zeros(image.shape[:2], dtype=np.uint8)
        #mask_right = mask_left.copy()
        mask_right = eye_on_mask(mask_right, right,shape=shape)
        #mask_right = eye_on_mask(mask_right, right)
        kernel = np.ones((9, 9), np.uint8)
        mask_right = cv2.dilate(mask_right, kernel, 5)
        #mask_right = cv2.dilate(mask_right, kernel, 5)
        eyes=cv2.bitwise_and(image,image,mask=mask_right)
        ret, thresh = cv2.threshold(mask_right, 163,255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE) #Rectangulo exterior con aproximacion simple
        for c in contours:
           
            x,y,w,h = cv2.boundingRect(c)
            crop_img = image[y:y+h, x:x+w]
            crop_img=brightness(crop_img,0.4)
            crop_img_gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
            image_enhanced = cv2.equalizeHist(crop_img_gray)
            image_enhanced=cv2.GaussianBlur(image_enhanced,(3,3),0)
            se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (7,7))
            bg=cv2.morphologyEx(image_enhanced, cv2.MORPH_DILATE, se)
            out_gray=cv2.divide(image_enhanced, bg, scale=255)
            out_binary=cv2.threshold(out_gray, 70, 255, cv2.THRESH_BINARY_INV )[1]
            contours1, hierarchy = cv2.findContours(out_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours1, key = cv2.contourArea)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(crop_img, (cX, cY), 1, (255, 255, 255), -1)        
            cv2.imshow("image", out_gray)
            cv2.imshow("sdsa", out_binary)
            
            cv2.imshow("ee", crop_img)
            cv2.imshow("asda", image)
            
    if cv2.waitKey(1) & 0xFF == ord('q'): # escape when q is pressed
        break

cv2.destroyAllWindows()
cap.release()