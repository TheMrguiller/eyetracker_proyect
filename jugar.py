from imutils import face_utils
import dlib
import cv2
import numpy as np
from helpers import *
import time


def compararPuntos(puntos, puntoLeido):

    i = 0
    distanciaMinima = 1000
    cuadranteElegido = 0

    print(len(puntos))

    for puntito in puntos:

        distancia = calcular_distancia(puntito,puntoLeido=puntoLeido)
        np.linalg.norm(np.array(puntos[i]) - np.array(puntoLeido))
        print(distancia)
        if distancia < distanciaMinima:
            cuadranteElegido = i
        i += 1
    
    return cuadranteElegido

def calcular_distancia(puntito,puntoLeido):
    righteyeratiowidth = puntito[3]/puntito[2]*puntoLeido[2].shape[1]
    righteyeratioheight = puntito[2]/puntito[3]*puntoLeido[2].shape[0]
    lefteyeratiowidth = puntito[8]/puntito[7]*puntoLeido[5].shape[1]
    lefteyeratioheight = puntito[7]/puntito[8]*puntoLeido[5].shape[0]
    righteyepoint = [puntito[0],puntito[1]]
    lefteyepoint = [puntito[4],puntito[5]]

#https://stackoverflow.com/questions/40800434/scale-resolution-of-image-keeping-point-locations



if __name__ == "__main__":
    p = "facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    eye_detector=eyeDetector()

    puntos = readTxt()
    print(puntos)

    while(True):
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
            if cX_r is not None or cX_l is not None :
                compararPuntos(puntos, puntosAct)
            else:
                print('Do nothing') ##mantener dibujado el ultimo

            if cv2.waitKey(1) & 0xFF == ord('q'): # escape when q is pressed
                break