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
    #print(puntoLeido)

    for puntito in puntos:

        distancia = calcular_distancia(puntito,puntoLeido=puntoLeido,seccion=i)
        
        #print(distancia)
        if distancia < distanciaMinima:
            distanciaMinima = distancia
            cuadranteElegido = i
            print(f"Seccion:{i} y distancia:{distancia}")
        i += 1


    #print(cuadranteElegido)
    return cuadranteElegido

def calcular_distancia(puntito,puntoLeido,seccion): ## recibo los dos puntos y tengo que escalarlos. 

    ###puntoEscalado = PuntoLeido * PguardadoSIZE / PuntoLeidoSIZE. Esto devuelve el punto correcto
    #print(puntoLeido[2])
    cX_r_escaled = (puntoLeido[0] * puntito[2]) / puntoLeido[2].shape[1]  ###DUDA ENTRE 0 o 1 en el shape Preguntar guille
    cY_r_escaled = (puntoLeido[1] * puntito[3]) / puntoLeido[2].shape[0]  ##x deberia ser ancho e y largo no??
    cX_l_escaled = (puntoLeido[3] * puntito[6]) / puntoLeido[5].shape[1]
    cY_l_escaled = (puntoLeido[4] * puntito[7]) / puntoLeido[5].shape[0]

    cX_r_saved = puntito[0]
    cY_r_saved = puntito[1]
    cX_l_saved = puntito[3]
    cY_l_saved = puntito[4]

    c_r = np.array([cX_r_saved, cY_r_saved]) 
    c_l = np.array([cX_l_saved, cY_l_saved])
    c_r_escaled = np.array([cX_r_escaled, cY_r_escaled])
    c_l_escaled = np.array([cX_l_escaled, cY_l_escaled])
     ##tenemos ya los puntos. Hay que compararlos por pares


    distanciaDerecha = calcularDistanciaEuclidea(c_r, c_r_escaled)
    distanciaIzquierda = calcularDistanciaEuclidea(c_l, c_l_escaled)
    #distancia = (distanciaDerecha + distanciaIzquierda ) / 2
    if seccion==0 or seccion==3 or seccion==6:
        return distanciaIzquierda
    elif seccion==1 or seccion==4 or seccion== 7:
        if distanciaDerecha >= distanciaIzquierda:
            return distanciaIzquierda
        else:
            return distanciaDerecha  
    else :
        return  distanciaDerecha

def calcularDistanciaEuclidea(ParPuntoXY, ParPuntoXY_actual):
    distanciaEntrePuntos = 1000
    ##calculamos la distancia euclide entre dos puntos
    distanciaEntrePuntos = np.linalg.norm(ParPuntoXY - ParPuntoXY_actual)

    return distanciaEntrePuntos


#https://stackoverflow.com/questions/40800434/scale-resolution-of-image-keeping-point-locations



if __name__ == "__main__":
    numberCuadrants = 9 #variable que recibiremos mediante funcion luego que nos dira el numero de cudarantes
    objetiveCuadrant = 0

    img = cv2.imread('imageTest.jpg')
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
            if cX_r is not None and cX_l is not None :
                objetiveCuadrant = compararPuntos(puntos, puntosAct)
                img=segmentador.paint_objective(image=img_copia,objetiveCuadrant=objetiveCuadrant)
                
            else:
                pass
                ##mantener dibujado el ultimo

            cv2.imshow('cuadrantes', img)
            if cv2.waitKey(1) & 0xFF == ord('q'): # escape when q is pressed
                break

###### Ideas de lo que puede ir mal##########
# Resize del ojo puede generar basura
# La distancia darle una vuelta mas, hacer por cuadrantes coger un ojo, coger solo un ojo
# La iluminacion puede afectar
# Probar con diferentes frames rate
# Probar con mi ordenador (Guiller)
# Darle alguna vuelta a la captura del ojo
# Probar la funcion de ajuste de puntos en vivo comparando con tu punto real