import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel, QApplication


def show_image(image_path='./screen.jpeg'):
    app = QApplication(sys.argv)
    pixmap = QtGui.QPixmap(image_path)
    screen = QLabel()
    screen.setScaledContents(True)
    screen.setPixmap(pixmap)
    screen.showFullScreen()
    sys.exit(app.exec_())


if __name__ == '__main__':
    import time
    time.sleep(3)
    show_image()