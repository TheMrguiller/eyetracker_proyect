#####este programa se encarga de pintar sobre una imagen 


import cv2
import numpy as np
from helpers import SegmentPinter



numberCuadrants = 9 #variable que recibiremos mediante funcion luego que nos dira el numero de cudarantes
objetiveCuadrant = 1

img = cv2.imread('imageTest.jpg')
segmentador=SegmentPinter(img,numberCuadrants)
segmentador.refresh_list()
image=segmentador.paint_cuadrants()
img=segmentador.paint_objective(image=image,objetiveCuadrant=objetiveCuadrant)
cv2.imshow('cuadrantes', img)


cv2.waitKey()