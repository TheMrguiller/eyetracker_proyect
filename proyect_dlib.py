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
#https://towardsdatascience.com/detecting-the-iris-and-changing-its-color-using-opencv-and-dlib-30a6aad122dd
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
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
left = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
right = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
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
            crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
            # Reduce the noise to avoid false circle detection
            ret, thresh = cv2.threshold(crop_img, 70,95, cv2.THRESH_BINARY)
            cnt, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
            for c1 in cnt:
                cv2.drawContours(crop_img, c1, -1, 0, 1)
        # show the output image with the face detections + facial landmarks
            cv2.imshow("Output", crop_img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # escape when q is pressed
        break

cv2.destroyAllWindows()
cap.release()