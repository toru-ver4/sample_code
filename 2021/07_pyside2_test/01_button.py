import sys
from PySide2.QtWidgets import QApplication, QPushButton
from PySide2.QtCore import Slot


# Greetings
@Slot()
def say_hello():
    print("Button clicked, Hello!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    button = QPushButton("Click me")
    button.clicked.connect(say_hello)
    button.show()
    app.exec_()
