# eyetracker_proyect
En este proyecto se ha creado un sistema capaz de identificar dónde está mirando el usuario y mostrarlo por pantalla. Para realizar este proyecto, se han utilizado herramientas puramente de visión, exceptuando aquellas que detectan la cara y la sección del ojo.
## Dependencias
Para poder poner en marcha el proyecto es necesario descargarse estas librerías.
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
## Información general
Este proyecto está formado principalmente por 4 elementos:
* aplicacionUi.py: Se trata de la aplicación desarrollada, en esta se observa tres elementos: jugar, calibrar y salir.
* calibrar.py : Se trata de la lógica encargada de calibrar el posicionamiento de los ojos para cada cuadrante.
* jugar.py: Se trata de la lógica principal del proyecto. En esta ventana, se pinta en qué dirección está mirando el ojo.
* helpers.py: En este fichero se encuentran las clases y funciones principales del proyecto.
* proyect_dlib_v2.py: En este script se puede observar los puntos calibrados respecto a la posición de nuestro iris.

## Puesta en marcha
Para poder utilizar la aplicación bastará con realizar la siguiente llamada.
```
python applicacionUi.py
```
Dentro de esta aplicación se observan tres opciones: calibrar,jugar y salir.
* Calibrar: Es la funcionalidad en la que se calibra el ojo. En la calibración se obtienen los 9 puntos de referencia. Estos puntos de referencia se utilizan para calcular la posición del ojo.
* Jugar: Es la funcionalidad principal. En esta se dibuja dónde está mirando el usuario.
* Salir: Se cierra la aplicación.
### Pantalla inicial
En la siguiente imagen se observa la pantalla principal de la aplicación.
![mainUI](https://user-images.githubusercontent.com/33113630/207388458-93484efd-a910-4184-9018-ced69e24e257.png)
### Calibración
En la calibración se irá obteniendo los diferentes puntos de referencia. A continuación se observan ciertos ejemplos.
![calibrar1](https://user-images.githubusercontent.com/33113630/207388602-73d953d5-9ebf-49cd-922f-807368acecdc.png)
![calibrar2](https://user-images.githubusercontent.com/33113630/207388615-1b05d63b-3b8e-4039-bc3c-09d06e006e0e.png)
![calibrar3](https://user-images.githubusercontent.com/33113630/207388625-17834f9a-6d17-4c20-8268-0d5c688cc53c.png)
### Jugar
En la siguiente imagen se observa la pantalla de juego.
![jugar](https://user-images.githubusercontent.com/33113630/207388658-8e9c3cc8-459a-466d-9fcd-0d3ce71740e9.png)
### Video de uso
https://user-images.githubusercontent.com/33113630/207388685-9d775482-7cd1-4c2b-b491-7bd6f366037d.mp4
## Problemas observados
En este proyecto se ha intentado dar solución a un problema mayor, la detección del iris sin usar tecnologías como cámaras con infrarrojo o machine learning. Durante el proceso de desarrollo se han encontrado grandes problemas a los que dar solución, siendo los siguientes:
* Iluminación: Al tratar la imagen mediante filtros, la iluminación afecta a la detección del iris. Para poder dar solución a este problema se ha detectado la iluminación y dependiendo de la calidad de la imagen se ha realizado un proceso u otro.
* Color de ojo: Las pruebas realizadas han sido principalmente con ojos marrones, aunque sí se han probado con ojos azules y verdes. Los resultados eran claramente mejores con los ojos marrones.
* Tamaño y forma de ojo: Dependiendo la forma del ojo, ojos más rasgados o ojos más saltones, la detección del ojo varía. Se han realizado varias pruebas con diferentes formas de ojo y los resultados han sido mejores de lo que se esperaba.
* Ojo humano: El ojo humano no cambia en gran medida de posición al mirar a ciertos puntos en el espacio. Al tener una periferia de visión bastante alta, no se necesita forzar mucho el ojo para ver en ciertos puntos del espacio.
* Calibración: Es necesario de un punto de referencia para poder identificar a qué lado está mirando el usuario. Aun así, a la hora de la calibración es necesario forzar algo la mirada, ya que los puntos se encuentran muy juntos entre sí.
## Trabajo futuro
Si se continuase con el proyecto, se entrenaría primero una inteligencia artificial para la detección de iris. Esto resolvería varios puntos importantes del proyecto como la iluminación,color de ojo, tamaño y forma. La calibración sería necesaria al no tener un punto de referencia con el que calcular las distancias.
## Extras
Si se quiere comprobar la posición del ojo respecto a los puntos de referencia, se puede ejecutar proyect_dlib_v2.py. Al ejecutar este archivo se observan los puntos de los ojos en conjunto con el punto actual del iris. En este programa, se puede observar claramente el mayor problema de la aplicación, el ojo humano.

https://user-images.githubusercontent.com/33113630/207835596-91d05cc2-d3fe-4621-a4fb-a666d74b1af8.mp4


