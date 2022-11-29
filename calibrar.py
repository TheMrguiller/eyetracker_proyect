import subprocess
from imutils import face_utils
import dlib
import cv2
import numpy as np
from helpers import *
import json
import os
from screeninfo import get_monitors



def get_screen_resolution():
    output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4',shell=True, stdout=subprocess.PIPE).communicate()[0]
    resolution = output.split()[0].split(b'x')
    return {'width': resolution[0], 'height': resolution[1]}

def onMouse(event, x, y, flags, param):
    global indice
    global copia
    if event == cv2.EVENT_LBUTTONDOWN:

        if indice >= 9:
            indice=0
        if x>=lista[indice][0][0] and x<= lista[indice][1][0] and y >= lista[indice][0][1] and y<= lista[indice][1][1]:
            indice += 1
            if indice >= 9:
                indice=0
            copia=cv2.rectangle(blank_image.copy(), lista[indice][0], lista[indice][1], (255,255,255), -1)
            file = open('config.txt', 'a')
            puntitos = get_positions(image=image,indice=indice)
            if puntitos is None:
                indice = indice - 1
                copia=cv2.rectangle(blank_image.copy(), lista[indice][0], lista[indice][1], (255,255,255), -1)
            file.write(puntitos+ '\n')
            file.close()
            
def create_points_list(height,width):
    seccion0= [(0,0),(50,50)]
    seccion1= [(int(width/2)-25,0),(int(width/2)+25,50)]
    seccion2= [(width-50,0),(width,50)]
    seccion3= [(0,int(height/2)-25),(50,int(height/2)+25)]
    seccion4= [(int(width/2)-25,int(height/2)-25),(int(width/2)+25,int(height/2)+25)]
    seccion5 = [(width-50,int(height/2)-25),(width,int(height/2)+25)]
    seccion6 = [(0,height-50),(50,height)]
    seccion7 = [(int(width/2)-25,height-50),(int(width/2)+25,height)]
    seccion8 = [(width-50,height-50),(width,height)]
    return [seccion0,seccion1,seccion2,seccion3,seccion4,seccion5,seccion6,seccion7,seccion8]

def get_positions(image,indice):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(indice)
    rects = detector(gray, 0)
    rect=get_frontal_face(rects)
    if rect ==0:
        pass
    else:
        rect=get_frontal_face(rects)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        cX_r,cY_r,crop_img_r,cX_l,cY_l,crop_img_l=eye_detector.get_eyes_position(image=image,shape=shape)
        if cX_r is not None and cX_l is not None:
            return write_txt(cX_r,cY_r,crop_img_r,cX_l,cY_l,crop_img_l)
        else:
            return None

def createTXT():
    f = open('config.txt', 'w')
    f.close()

def write_txt(cX_r,cY_r,crop_img_r,cX_l,cY_l,crop_img_l): 
    cadenaCompleta = "cX_r:" + str(cX_r) + ',' + "cY_r:" + str(cY_r) + ',' + "w_r:" + str(crop_img_r.shape[1]) + ',' + "h_r:" + str(crop_img_r.shape[0]) + ',' + "cX_l:" + str(cX_l) + ',' + "cY_l:" + str(cY_l) + ','"w_l:" + str(crop_img_l.shape[1]) + ',' +  "h_l:" + str(crop_img_l.shape[0])
    return cadenaCompleta

def readTxt():

    lineasPunto =  []
    with open('config.txt') as f:
        lines = f.readlines()
        for line in lines:
            lineaUnica = []
            line = line.strip()
            splited = line.split(",")
            #print(splited)
            for tupla in splited:
                #print(tupla)
                tupla = tupla.split(":")
                #print(tupla[1])  ##todos los valores de una linea
                lineaUnica.append(int(tupla[1]))
            lineasPunto.append(lineaUnica)
        print(lineasPunto)
        f.close()
        return lineasPunto


if __name__ == "__main__":

    indice=0

    width=1920 
    height=1080
    for m in get_monitors():
        width = m.width
        height = m.height

    width = int(width * 0.91)
    height = int(height * 0.91)
    size = get_screen_resolution()
    blank_image = np.zeros((height,width,3), np.uint8)
    copia = blank_image.copy()
    height = blank_image.shape[0]
    width = blank_image.shape[1]
    file = createTXT()
    lista=create_points_list(height=height,width=width)
    cv2.rectangle(copia, lista[0][0], lista[0][1], (255,255,255), -1)
    cv2.namedWindow('Calibrar')
    cv2.setMouseCallback('Calibrar', onMouse)
    p = "facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    eye_detector=eyeDetector()
    while True:
        _, image = cap.read()
        cv2.imshow("Calibrar",copia)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): # escape when q is pressed
            readTxt()
            break
            
    cv2.destroyAllWindows()
    cap.release()
