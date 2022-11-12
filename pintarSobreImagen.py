#####este programa se encarga de pintar sobre una imagen 


import cv2
import numpy as np




numberCuadrants = 4 #variable que recibiremos mediante funcion luego que nos dira el numero de cudarantes
objetiveCuadrant = 1
N = np.sqrt(numberCuadrants)
print(N)
img = cv2.imread('imageTest.jpg')
width = img.shape[1]  ##solo cogemos el height y la width
height = img.shape[0]
print(height)
print(width)

heightInterval = height//N
withInterval = width//N

print(heightInterval)
print(withInterval)

listaCuadrants = []

for i in range(int(N)):
    iwi = i*withInterval
    ihe = i*heightInterval


    for j in range(int(N)):
        punto1 = (int((withInterval*j)), int((heightInterval*i)))
        #punto2 = (int((withInterval*(j+1)) + iwi), int((heightInterval*(i+1)) + ihe))
        punto2 = (int(punto1[0] + withInterval), int(punto1[1] + heightInterval))
        listaPuntos = []
        listaPuntos.append(punto1)
        listaPuntos.append(punto2)
        print(listaPuntos)
        listaCuadrants.append(listaPuntos)
    #me recorro el for 9 vece para introducir cada uno de los cuadrantes 


for i in range(int(N) - 1):
    i += 1 ##Ya corregire esto luego es para ir haciendo pruebas (y, x)
    print((int(heightInterval * i)))
    image = cv2.line(img, (0, int(heightInterval * i)), (width, int(heightInterval * i)), (0,0,0), 5)
    image = cv2.line(img, (int(withInterval * i), 0 ), (int(withInterval*i), height), (0,0,0), 5)

    
print(listaCuadrants[objetiveCuadrant][0])
print(listaCuadrants[objetiveCuadrant][1])
image = cv2.line(img, listaCuadrants[objetiveCuadrant][0], listaCuadrants[objetiveCuadrant][1], (0,0,0), 5)

cv2.imshow('cuadrantes', img)


cv2.waitKey()