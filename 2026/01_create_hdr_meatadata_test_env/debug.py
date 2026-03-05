import os

import numpy as np
from screeninfo import get_monitors
# from screeninfo.common import Monitor


def debug_screen_info():
    for m in get_monitors():
        print(str(m))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_screen_info()
