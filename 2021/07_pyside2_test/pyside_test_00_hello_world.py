import sys
from PySide2.QtWidgets import QApplication, QLabel


if __name__ == "__main__":
    app = QApplication(sys.argv)
    label = QLabel("<font color=red size=40>Hello World!</font>")
    label.show()
    app.exec_()
