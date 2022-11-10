import cv2
import dlib
import numpy as np

#https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6
#https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
    

vid = cv2.VideoCapture(0) 

while(True):

    ret, frame = vid.read()
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.02,10)
    rects=[]
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        predictor = dlib.shape_predictor('facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat')
        rects.append((x, y, w, h))
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        for (xs, ys) in shape:
            cv2.circle(img, (xs, ys), 2, (0, 0, 255), -1)

    cv2.namedWindow('eye')
    cv2.imshow('eye', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # 16 bits mayuscula 8 bits minuscula
        break;
vid.release()# Soltamos el objeto captura
cv2.destroyAllWindows() # Destruimos todas las ventanas generadas