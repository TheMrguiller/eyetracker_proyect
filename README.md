# eyetracker_proyect
En este proyecto se ha creado un sistema capz de identificar donde esta mirando el usuario y pintarlo en pantalla. Para realizar este proyecto se han utilizado herramientas puramente de vision, exceptuando aquellas que detectan la cara y la seccion del ojo.
## Dependencias 
Para poder poner en marcha el proyecto es necesario descargarse unas librerias basicas.
```
pip install cmake
pip install dlib
pip install imutils
pip install PyQt5
sudo apt-get install qttools5-dev-tools
pip install screeninfo
```
## Puesta en marcha
```
python applicacionUi.py
```
## Informacion general
Ese proyecto esta formado principalment por 4 elementos:
* aplicacionUi.py: Se trata de la aplicacion desarrollada, en esta se observa tres elementos: jugar, calibrar y salir
* calibrar.py : Se trata de la logica encargada de calibrar el posicionamiento de los ojos para cada cuadrante.
* jugar.py: Se trata de la logica principal del proyecto. En esta ventana, se pinta en que direccion esta mirando el ojo.
* helpers.py: En este fichero se encuentras las clases y funciones principales del proyecto

Para salir de cada una de las pantallas bastaria con cerrar la ventana o presionar la letra s de salir. En este proyecto se ha intentado dar solucion a un problema mayor, la deteccion del iris sin usar tecnologias como camaras con infrarrojo o machine learning. Durante el proceso de desarrolo se han encontrado grandes problemas a los que dar solucion, siendo los siguientes:
* Iluminacion: Al tratar la imagen mediante filtros, la iluminacion afecta a la deteccion del iris. Para poder dar solucion a este problema se ha detectado la iluminacion y dependiendo de la calidad de la imagen se ha realizado un proceso u otro.
* Color de ojo: Las pruebas realizadas han sido principalmente con ojos marrones, aunque si se han probado con ojos azules y verdes. Los resultados eran claramente mejores con los ojos marrones.
* Tamaño y forma de ojo: Dependiendo la forma del ojo, ojos mas rasgados o ojos mas saltones, la deteccion del ojo varia. Se han realizado varias pruebas con diferentes formas de ojo y los resultados han sido mejores de lo que se esperaba.
* Ojo humano: El ojo humano no cambia en gran medida de posicion al mirar a ciertos puntos en el espacio. Al tener una periferia de vision bastante alta, no se necesita forzar mucho el ojo para ver en ciertos puntos del espacio.
* Calibracion: Es necesario de un punto de referencia para poder identificar a que lado esta mirando el usuario. Aun asi, a la hora de la calibracion es necesario forzar algo la mirada, ya que los puntos se encuentran muy juntos entre si.
## Trabajo futuro
Si se continuase con el proyecto, se entrenaria primero una inteligencia artificial para la deteccion de iris. Esto resolveria varios puntos importantes del proyecto como la iluminacion,color de ojo, tamaño y forma. La calibracion seria siendo necesaria al no tener un punto de de referencia con el que calcular las distancias.

