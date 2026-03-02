from pathlib import Path
import os

import numpy as np
from imagecodecs import imread

import test_pattern_generator2 as tpg


def debug_diff():
    ref_image = imread("./capture_scRGB/img/dst_windows_official_screenshot.jxr")
    test_image = imread("./capture_scRGB/img/dst_capture_scRGB_screenshot.jxr")

    # diff = np.abs(ref_image - test_image)
    # diff = diff / np.max(diff)

    # tpg.img_wirte_float_as_16bit_int("./debug.png", diff)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # tpg.jxr_to_exr("./capture_scRGB/hoge.jxr")
    debug_diff()
