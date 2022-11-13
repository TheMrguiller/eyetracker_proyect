from imutils import face_utils
import dlib
import cv2
import numpy as np
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
def eye_on_mask(mask, side):
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

def blob_process(img, detector):
    se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    bg=cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
    out_gray=cv2.divide(img, bg, scale=255)
    out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY )[1]     
    keypoints = detector.detect(out_binary)
    return keypoints
def nothing(x):
    pass
def brightness(image,gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(image, lookUpTable)
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector_pupil = cv2.SimpleBlobDetector_create(detector_params)
left = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
right = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye
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
        # loop over the face detections and get the nearest one
        rect=get_frontal_face(rects)
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        #for (x, y) in shape:
        #    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        mask_left = np.zeros(image.shape[:2], dtype=np.uint8)
        #mask_right = mask_left.copy()
        mask_left = eye_on_mask(mask_left, left)
        #mask_right = eye_on_mask(mask_right, right)
        kernel = np.ones((9, 9), np.uint8)
        mask_left = cv2.dilate(mask_left, kernel, 5)
        #mask_right = cv2.dilate(mask_right, kernel, 5)
        eyes=cv2.bitwise_and(image,image,mask=mask_left)
        ret, thresh = cv2.threshold(mask_left, 163,255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE) #Rectangulo exterior con aproximacion simple
        for c in contours:
            # encontrar las coordenadas del cuadrado delimitador
            x,y,w,h = cv2.boundingRect(c)
            crop_img = image[y:y+h, x:x+w]
            crop_img=brightness(crop_img,0.4)
            crop_img_gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
            image_enhanced = cv2.equalizeHist(crop_img_gray)
            image_enhanced=cv2.GaussianBlur(image_enhanced,(3,3),0)
            se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (7,7))
            bg=cv2.morphologyEx(image_enhanced, cv2.MORPH_DILATE, se)
            out_gray=cv2.divide(image_enhanced, bg, scale=255)
            out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1]
            thresholded, contours = cv2.findContours(out_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
        
        
            """keypoints=blob_process(image_enhanced,detector_pupil)
            print(keypoints)
            cv2.drawKeypoints(crop_img, keypoints, crop_img, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)"""
            cv2.imshow("image", out_gray)
            cv2.imshow("sdsa", out_binary)
            cv2.imshow("ee", crop_img)
            #cv2.imshow("Modificacion", binary)
    if cv2.waitKey(1) & 0xFF == ord('q'): # escape when q is pressed
        break

cv2.destroyAllWindows()
cap.release()