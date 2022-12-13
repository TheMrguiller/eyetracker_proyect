# eyetracker_proyect
En este proyecto se ha creado un sistema capaz de identificar donde esta mirando el usuario y pintarlo en pantalla. Para realizar este proyecto, se han utilizado herramientas puramente de vision, exceptuando aquellas que detectan la cara y la seccion del ojo.
## Dependencias 
Para poder poner en marcha el proyecto es necesario descargarse estas librerias.
```
pip install cmake
pip install dlib
pip install imutils
pip install PyQt5
sudo apt-get install qttools5-dev-tools
pip install screeninfo
pip install numpy
pip install opencv-python
```
## Informacion general
Este proyecto esta formado principalmente por 4 elementos:
* aplicacionUi.py: Se trata de la aplicacion desarrollada, en esta se observa tres elementos: jugar, calibrar y salir
* calibrar.py : Se trata de la logica encargada de calibrar el posicionamiento de los ojos para cada cuadrante.
* jugar.py: Se trata de la logica principal del proyecto. En esta ventana, se pinta en que direccion esta mirando el ojo.
* helpers.py: En este fichero se encuentras las clases y funciones principales del proyecto

## Puesta en marcha
Para poder utilizar la aplicacion bastaria realizar la siguiente llamada.
```
python applicacionUi.py
```
Dentro de esta aplicacion se observaran tres opciones: calibrar,jugar y salir.
* Calibrar: Es la funcionalidad en la que se calibra el ojo. En esta calibracion se obtienen los 9 puntos de referencia. Estos puntos de referencia se utilizan para calcular la posicion del ojo.
* Jugar: Es la funcionalidad principal. En esta se dibuja donde esta mirando el usuario.
* Salir: Se cierra la aplicacion.
### Pantalla inicial
En la siguiente imagen se observa la pantalla principal de la aplicacion.
![mainUI](https://user-images.githubusercontent.com/33113630/207388458-93484efd-a910-4184-9018-ced69e24e257.png)
### Calibración
En la calibracion se ira obteniendo los diferentes puntos de referencia. A continuacion se observan ciertos ejemplos.
![calibrar1](https://user-images.githubusercontent.com/33113630/207388602-73d953d5-9ebf-49cd-922f-807368acecdc.png)
![calibrar2](https://user-images.githubusercontent.com/33113630/207388615-1b05d63b-3b8e-4039-bc3c-09d06e006e0e.png)
![calibrar3](https://user-images.githubusercontent.com/33113630/207388625-17834f9a-6d17-4c20-8268-0d5c688cc53c.png)
### Jugar
En la siguiente imagen se observa la pantalla de juego.
![jugar](https://user-images.githubusercontent.com/33113630/207388658-8e9c3cc8-459a-466d-9fcd-0d3ce71740e9.png)
### Video de ejemplo
https://user-images.githubusercontent.com/33113630/207388685-9d775482-7cd1-4c2b-b491-7bd6f366037d.mp4

## Problemas observados
En este proyecto se ha intentado dar solucion a un problema mayor, la deteccion del iris sin usar tecnologias como camaras con infrarrojo o machine learning. Durante el proceso de desarrolo se han encontrado grandes problemas a los que dar solucion, siendo los siguientes:
* Iluminacion: Al tratar la imagen mediante filtros, la iluminacion afecta a la deteccion del iris. Para poder dar solucion a este problema se ha detectado la iluminacion y dependiendo de la calidad de la imagen se ha realizado un proceso u otro.
* Color de ojo: Las pruebas realizadas han sido principalmente con ojos marrones, aunque si se han probado con ojos azules y verdes. Los resultados eran claramente mejores con los ojos marrones.
* Tamaño y forma de ojo: Dependiendo la forma del ojo, ojos mas rasgados o ojos mas saltones, la deteccion del ojo varia. Se han realizado varias pruebas con diferentes formas de ojo y los resultados han sido mejores de lo que se esperaba.
* Ojo humano: El ojo humano no cambia en gran medida de posicion al mirar a ciertos puntos en el espacio. Al tener una periferia de vision bastante alta, no se necesita forzar mucho el ojo para ver en ciertos puntos del espacio.
* Calibracion: Es necesario de un punto de referencia para poder identificar a que lado esta mirando el usuario. Aun asi, a la hora de la calibracion es necesario forzar algo la mirada, ya que los puntos se encuentran muy juntos entre si.
## Trabajo futuro
Si se continuase con el proyecto, se entrenaria primero una inteligencia artificial para la deteccion de iris. Esto resolveria varios puntos importantes del proyecto como la iluminacion,color de ojo, tamaño y forma. La calibracion seria siendo necesaria al no tener un punto de de referencia con el que calcular las distancias.
## Extras
Si se quiere comprobar la posicion del ojo respeto a los puntos de referencia, se puede ejecutar proyect_dlib_v2.py. Al ejecutar este archivo se observan los puntos de los ojos enconjunto con el punto actual del iris. En este programa, se puede observar claramente el mayor problema de la aplicacion, el ojo humano.
