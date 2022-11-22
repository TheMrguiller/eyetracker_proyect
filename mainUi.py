from PyQt5 import QtCore, QtGui, QtWidgets
import os

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(754, 245)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 90, 171, 91))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.funcionBoton1)



        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(300, 90, 171, 91))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.funcionBoton2)



        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(540, 90, 171, 91))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.funcionBoton3)



        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Tracker"))
        self.pushButton_2.setText(_translate("MainWindow", "Calibrar"))
        self.pushButton_3.setText(_translate("MainWindow", "Salir"))


    def funcionBoton1(self):
        os.system("python pintarSobreImagen_v2.py")
        print("Has clicado al boton1")

    def funcionBoton2(self):
        os.system("python calibrar.py")
        print("Has clicado al boton2")

    def funcionBoton3(self):
        print("Has clicado al boton3")
        self.close() 


