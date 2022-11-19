from imutils import face_utils
import dlib
import cv2
import numpy as np


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
    #Inicializamos las variables que no cambian en la ejecucion
    def __init__(self):
        self.right = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
        self.left = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye
        self.kernel = np.ones((9, 9), np.uint8)
    #Devuelve las mascaras de cada ojo
    def eye_mask_generator(self):
        mask_right=np.zeros(self.image.shape[:2], dtype=np.uint8)
        mask_left=np.zeros(self.image.shape[:2], dtype=np.uint8)
        mask_right = self.eye_on_mask(mask_right, self.right)
        mask_left = self.eye_on_mask(mask_left, self.left)
        return mask_right,mask_left
    #Funcion que devuelve la posicion de los ojos
    def get_eyes_position(self,image,shape):
        self.image=image
        self.shape=shape
        mask_right,mask_left=self.eye_mask_generator()
        cX_r,cY_r,crop_img_r=self.get_position(mask_right)
        cX_l,cY_l,crop_img_l=self.get_position(mask_left)
        return cX_r,cY_r,crop_img_r,cX_l,cY_l,crop_img_l
    #Funcion que devuelve la posicion del ojo dado una mascara
    def get_position(self,mask):
        mask = cv2.dilate(mask, self.kernel, 5)#Aumentamos el tamaño de la mascara
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE) #Rectangulo exterior con aproximacion simple
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
    #Calcula el centro del contorno
    def calculate_center(self,countour):
        M = cv2.moments(countour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX,cY
    #Genera las mascaras de cada ojo   
    def eye_on_mask(self,mask, side):
        points = [self.shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)
        return mask
    #aumenta el brillo de la imagen, en nuestro caso la que solo muestra el ojo
    def brightness(self,image,gamma=0.4):
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        return cv2.LUT(image, lookUpTable)