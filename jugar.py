from imutils import face_utils
import dlib
import cv2
import numpy as np
from helpers import *
import time
from screeninfo import get_monitors
#https://stackoverflow.com/questions/40800434/scale-resolution-of-image-keeping-point-locations



if __name__ == "__main__":
    numberCuadrants = 9 #variable que recibiremos mediante funcion luego que nos dira el numero de cudarantes
    objetiveCuadrant = 0
    for m in get_monitors():
        width = m.width
        height = m.height

    width = int(width * 0.91)
    height = int(height * 0.91)
    img = np.zeros((height,width,3), np.uint8)
    img.fill(255)
    #img = cv2.imread('imageTest.jpg')
    img_copia= img.copy()
    segmentador=SegmentPinter(img_copia,numberCuadrants)
    segmentador.refresh_list()
    img_copia=segmentador.paint_cuadrants()
    p = "facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    eye_detector=eyeDetector()

    puntos = readTxt()
    #print(puntos)
   
    while True:
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

            puntosAct = [cX_r,cY_r,crop_img_r,cX_l,cY_l,crop_img_l]
            
            #end_time_ns = time.process_time_ns()
            #timer_ns = end_time_ns - start_time_ns
            #print(timer_ns)
            if cX_r is not None and cX_l is not None :
                objetiveCuadrant = compararPuntos(puntos, puntosAct)
                img=segmentador.paint_objective(image=img_copia,objetiveCuadrant=objetiveCuadrant)
                
            else:
                pass
                ##mantener dibujado el ultimo

            cv2.imshow('cuadrantes', img)

            if cv2.waitKey(1) & 0xFF == ord('s'): # escape when q is pressed
                break
            if cv2.getWindowProperty('cuadrantes',cv2.WND_PROP_VISIBLE) < 1:        
                break  

cv2.destroyAllWindows()
cap.release()

###### Ideas de lo que puede ir mal##########
# Resize del ojo puede generar basura. check
# La distancia darle una vuelta mas, hacer por cuadrantes coger un ojo, coger solo un ojo. Check
# La iluminacion puede afectar. check
# Probar con diferentes frames rate
# Probar con mi ordenador (Guiller)
# Darle alguna vuelta a la captura del ojo
# Probar la funcion de ajuste de puntos en vivo comparando con tu punto real. check

#### TO DO #######
# calibrar termine cuando llegue al ultimo punto, que se pueda cerrar con la x. Check
# Jugar sea con una pantalla del mismo tamaño que la otra, que sea blanca y que se pueda cerrar con la x. Check
# Limpiar codigo y carpetas. Falta