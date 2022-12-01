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

    #print(len(puntos))
    print(puntoLeido)

    for puntito in puntos:

        distancia = calcular_distancia(puntito,puntoLeido=puntoLeido)
        np.linalg.norm(np.array(puntos[i]) - np.array(puntoLeido))
        #print(distancia)
        if distancia < distanciaMinima:
            cuadranteElegido = i
        i += 1
    
    return cuadranteElegido

def calcular_distancia(puntito,puntoLeido): ## recibo los dos puntos y tengo que escalarlos. 

    ###puntoEscalado = PuntoLeido * PguardadoSIZE / PuntoLeidoSIZE. Esto devuelve el punto correcto
    cX_r_escaled = (puntoLeido[0] * puntito[2].shape[0]) / puntoLeido[2].shape[0]  ###DUDA ENTRE 0 o 1 en el shape Preguntar guille
    cY_r_escaled = (puntoLeido[1] * puntito[2].shape[1]) / puntoLeido[2].shape[1]  ##x deberia ser ancho e y largo no??
    cX_l_escaled = (puntoLeido[3] * puntito[5].shape[0]) / puntoLeido[5].shape[0]
    cY_l_escaled = (puntoLeido[4] * puntito[5].shape[1]) / puntoLeido[5].shape[1]

    cX_r_saved = puntito[0]
    cY_r_saved = puntito[1]
    cX_l_saved = puntito[3]
    cY_l_saved = puntito[4]

    c_r = (cX_r_saved, cY_r_saved) 
    c_l = (cX_l_saved, cY_l_saved)
    c_r_escaled = (cX_r_escaled, cY_r_escaled)
    c_l_escaled = (cX_l_escaled, cY_l_escaled)
     ##tenemos ya los puntos. Hay que compararlos por pares


    distanciaDerecha = calcularDistanciaEuclidea(c_r, c_r_escaled)
    distanciaIzquierda = calcularDistanciaEuclidea(c_l, c_l_escaled)

    if distanciaDerecha >= distanciaIzquierda:
        return distanciaIzquierda
    else:
        return distanciaDerecha   

def calcularDistanciaEuclidea(ParPuntoXY, ParPuntoXY_actual):
    distanciaEntrePuntos = 1000
    ##calculamos la distancia euclide entre dos puntos
    return distanciaEntrePuntos


#https://stackoverflow.com/questions/40800434/scale-resolution-of-image-keeping-point-locations



if __name__ == "__main__":
    p = "facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    eye_detector=eyeDetector()

    puntos = readTxt()
    #print(puntos)

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