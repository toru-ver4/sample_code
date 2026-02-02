from pathlib import Path
import os

import numpy as np

import test_pattern_generator2 as tpg


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # tpg.jxr_to_exr("./debug/png_debug.jxr")
    tpg.jxr_to_exr("./debug/mp4_debug2.jxr")
