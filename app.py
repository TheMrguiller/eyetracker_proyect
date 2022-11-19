#https://pythonpyqt.com/pyqt-window/
import sys
from PyQt5 import QtWidgets,QtGui


app = QtWidgets.QApplication(sys.argv)
app.setWindowIcon(QtGui.QIcon("logo.png"))
windows = QtWidgets.QWidget()
screen_resolution = app.desktop().screenGeometry()
width, height = screen_resolution.width(), screen_resolution.height()
windows.resize(width,height)
windows.show()
windows.setWindowTitle('EyeTracker')

sys.exit(app.exec_())