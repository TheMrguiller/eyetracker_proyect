import sys


from PyQt5.QtWidgets import (

    QApplication, QDialog, QMainWindow, QMessageBox

)

from PyQt5.uic import loadUi


from mainUi import Ui_MainWindow


class FindReplaceDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        loadUi("ui/find_replace.ui", self)


class Window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setupUi(self)


    def findAndReplace(self):

        dialog = FindReplaceDialog(self)

        dialog.exec()


    def about(self):

        QMessageBox.about(

            self,

            "About Sample Editor",

            "<p>A sample text editor app built with:</p>"

            "<p>- PyQt</p>"

            "<p>- Qt Designer</p>"

            "<p>- Python</p>",

        )



if __name__ == "__main__":

    app = QApplication(sys.argv)

    win = Window()

    win.show()

    sys.exit(app.exec())