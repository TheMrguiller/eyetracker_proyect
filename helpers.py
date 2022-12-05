from imutils import face_utils,grab_contours
import cv2
import numpy as np
from numpy.linalg import norm

def compararPuntos(puntos, puntoLeido):

    i = 0
    distanciaMinima = 1000
    cuadranteElegido = 0

    for puntito in puntos:
        distancia = calcular_distancia(puntito,puntoLeido=puntoLeido,seccion=i)  
        if distancia < distanciaMinima:
            distanciaMinima = distancia
            cuadranteElegido = i
        i += 1

    return cuadranteElegido

def calcular_distancia(puntito,puntoLeido,seccion):

    
    cX_r_escaled = int(puntito[0] * (puntoLeido[2].shape[1] / puntito[2]  ) )
    cY_r_escaled = int(puntito[1] * (puntoLeido[2].shape[0] / puntito[3] ) ) 
    cX_l_escaled = int(puntito[4] * (puntoLeido[5].shape[1] / puntito[6] ))
    cY_l_escaled = int(puntito[5] * (puntoLeido[5].shape[0] / puntito[7]  ))

    cX_r_saved = puntito[0]
    cY_r_saved = puntito[1]
    cX_l_saved = puntito[3]
    cY_l_saved = puntito[4]

    c_r = np.array([cX_r_saved, cY_r_saved]) 
    c_l = np.array([cX_l_saved, cY_l_saved])
    c_r_escaledp = np.array([cX_r_escaled, cY_r_escaled])
    c_l_escaledp = np.array([cX_l_escaled, cY_l_escaled])
    
    distanciaDerecha = calcularDistanciaEuclidea(c_r, c_r_escaledp)
    distanciaIzquierda = calcularDistanciaEuclidea(c_l, c_l_escaledp)
   
    if distanciaDerecha >= distanciaIzquierda:
        return distanciaIzquierda
    else:
        return distanciaDerecha  

def calcularDistanciaEuclidea(ParPuntoXY, ParPuntoXY_actual):
    
    distanciaEntrePuntos = np.linalg.norm(ParPuntoXY - ParPuntoXY_actual)

    return distanciaEntrePuntos

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
        f.close()
        return lineasPunto


def eye_on_mask(mask, side,shape):
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

def brightness(image,gamma=0.4):
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        return cv2.LUT(image, lookUpTable)

class eyeDetector:

   
    def __init__(self):
        self.right = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
        self.left = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye
        self.kernel = np.ones((9, 9), np.uint8)
    
    def eye_mask_generator(self):
        mask_right=np.zeros(self.image.shape[:2], dtype=np.uint8)
        mask_left=np.zeros(self.image.shape[:2], dtype=np.uint8)
        mask_right = self.from_mask_points_get_mask(mask_right, self.right)
        mask_left = self.from_mask_points_get_mask(mask_left, self.left)
        return mask_right,mask_left
    #Funcion que devuelve la posicion de los ojos
    def get_eyes_position(self,image,shape):
        self.image=image
        self.shape=shape
        mask_right,mask_left=self.eye_mask_generator()
        cX_r,cY_r,crop_img_r=self.get_positon_v2(mask_right)
        cX_l,cY_l,crop_img_l=self.get_positon_v2(mask_left)
        return cX_r,cY_r,crop_img_r,cX_l,cY_l,crop_img_l

    #Funcion que devuelve la posicion del ojo dado una mascara, version antigua.
    def get_position(self,mask):
        mask = cv2.dilate(mask, self.kernel, 5)
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE) 
        c=contours[0]
        x,y,w,h = cv2.boundingRect(c)
        crop_img = self.image[y:y+h, x:x+w]
        crop_img=self.brightness(crop_img,0.4)
        crop_img_gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
        image_enhanced = cv2.equalizeHist(crop_img_gray)
        image_enhanced=cv2.GaussianBlur(image_enhanced,(3,3),0)
        se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (7,7))
        bg=cv2.morphologyEx(image_enhanced, cv2.MORPH_DILATE, se)
        out_gray=cv2.divide(image_enhanced, bg, scale=255)
        out_binary=cv2.threshold(out_gray, 70, 255, cv2.THRESH_BINARY_INV )[1]
        contours1, hierarchy = cv2.findContours(out_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours1, key = cv2.contourArea)
        cX,cY=self.calculate_center(c)
        return cX,cY,crop_img

    #Funcion que devuelve la posicion del ojo dado una mascara, version mejorada.
    def get_positon_v2(self,eye_mask):
        dilated_mask = cv2.dilate(eye_mask, self.kernel, 1)
        
        contours, hier = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE) #Rectangulo exterior con aproximacion simple
        max_c=contours[0]
        
        x,y,w,h = cv2.boundingRect(max_c)
        
        crop_img = self.image[y:y+h, x:x+w]
        
        brightness_value=self.brightness_calculator(crop_img)
        if brightness_value < 25:
            crop_img=self.change_brightness_v2(crop_img,0.4)    
        crop_img_gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
        image_enhanced = cv2.equalizeHist(crop_img_gray)
        image_enhanced=cv2.GaussianBlur(image_enhanced,(3,3),0)
        se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (7,7))
        bg=cv2.morphologyEx(image_enhanced, cv2.MORPH_DILATE, se)
        out_gray=cv2.divide(image_enhanced, bg, scale=255)
        
        image_binary=cv2.threshold(out_gray, 120, 255, cv2.THRESH_BINARY_INV )[1]
        
        contours, hier = cv2.findContours(image_binary, cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE)
        eye_mask=np.zeros(crop_img.shape[:2], dtype=np.uint8)
        max_c = max(contours, key = cv2.contourArea)
        eye_mask=cv2.drawContours(eye_mask, [max_c], -1, color=255, thickness=cv2.FILLED)
        
        kernel = np.ones((3,3),np.uint8)
        eye_mask_erode = cv2.erode(eye_mask,kernel,iterations = 2)
        contours, hier = cv2.findContours(eye_mask_erode, cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE)
        if not contours:
            pass
        else:
            max_c = max(contours, key = cv2.contourArea)
            eye_mask=np.zeros(crop_img.shape[:2], dtype=np.uint8)
            eye_mask_erode=cv2.drawContours(eye_mask, [max_c], -1, color=255, thickness=cv2.FILLED)
            eye_mask_opening = cv2.morphologyEx(eye_mask_erode, cv2.MORPH_OPEN, kernel)
            eye_mask_dilate = cv2.dilate(eye_mask_opening,kernel,iterations = 1)
            
            final_contour, hierarchy = cv2.findContours(eye_mask_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if not final_contour:
                pass
            else:
                max_c = max(final_contour, key = cv2.contourArea)
                cX,cY=self.calculate_center(max_c)
                #cv2.circle(crop_img, (cX, cY), 1, (255, 255, 255), -1)  
                #cv2.imshow("ojo_izquierdo",crop_img)
                return cX,cY,crop_img
        return None,None,crop_img

    #Calcula el centro del contorno
    def calculate_center(self,countour):
        M = cv2.moments(countour)
        if M['m00'] != 0:
            cX = int(M['m10']/M['m00'])
            cY = int(M['m01']/M['m00'])
        return cX,cY

    #Genera las mascaras de cada ojo   
    def from_mask_points_get_mask(self,mask, side):
        points = [self.shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)
        return mask

    #Change brigthness of image
    def change_brightness(self,img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v,value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def brightness_calculator(self,img):
        if len(img.shape) == 3:
            # Colored RGB or BGR (*Do Not* use HSV images with this function)
            # create brightness with euclidean norm
            return np.average(norm(img, axis=2)) / np.sqrt(3)
        else:
            # Grayscale
            return np.average(img)

    #aumenta el brillo de la imagen, en nuestro caso la que solo muestra el ojo
    def change_brightness_v2(self,image,gamma=0.4):
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        return cv2.LUT(image, lookUpTable)

class SegmentPinter:
    def __init__(self,image,numberCuadrants=9):
        self.numberCuadrants = numberCuadrants
        self.N = np.sqrt(self.numberCuadrants)
        self.image=image
        self.width = self.image.shape[1]  ##solo cogemos el height y la width
        self.height = self.image.shape[0]
        self.heightInterval = self.height//self.N
        self.withInterval = self.width//self.N
        self.listaCuadrants = []
    #Reiniciamos nuestra matriz de puntos
    def refresh_list(self):
        self.listaCuadrants.clear()
        for i in range(int(self.N)):
            #iwi = i*self.withInterval
            #ihe = i*self.heightInterval
            for j in range(int(self.N)):
                punto1 = (int((self.withInterval*j)), int((self.heightInterval*i)))
                #punto2 = (int((withInterval*(j+1)) + iwi), int((heightInterval*(i+1)) + ihe))
                punto2 = (int(punto1[0] + self.withInterval), int(punto1[1] + self.heightInterval))
                listaPuntos = []
                listaPuntos.append(punto1)
                listaPuntos.append(punto2)
                self.listaCuadrants.append(listaPuntos) 
    #Pintamos la matriz de cuadrantes en la imagen
    def paint_cuadrants(self):
        image= self.image.copy()
        for i in range(int(self.N) - 1):
            i += 1 ##Ya corregire esto luego es para ir haciendo pruebas (y, x)
            image = cv2.line(image, (0, int(self.heightInterval * i)), (self.width, int(self.heightInterval * i)), (0,0,0), 5)
            image = cv2.line(image, (int(self.withInterval * i), 0 ), (int(self.withInterval*i), self.height), (0,0,0), 5)
        return image
    #Pintamos el objtivo
    def paint_objective(self,image,objetiveCuadrant):
        image=self.tranparetColor(self.listaCuadrants[objetiveCuadrant][0],self.listaCuadrants[objetiveCuadrant][1],image)
        #image = cv2.line(image, self.listaCuadrants[objetiveCuadrant][0], self.listaCuadrants[objetiveCuadrant][1], (0,0,0), 5)
        return image
    def tranparetColor(self,x_point,x1_point,image):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(mask, x_point, x1_point, 255, -1)
        trans_image = np.copy(image)
        trans_image[(mask==255)] = [0,255,0]
        return cv2.addWeighted(trans_image, 0.3, image, 0.7, 0, trans_image)
