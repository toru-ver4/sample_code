from pathlib import Path
import os

import numpy as np

import test_pattern_generator2 as tpg


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    tpg.jxr_to_exr("./capture_scRGB/hoge.jxr")
