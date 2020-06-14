# -*- coding: utf-8 -*-
"""
debug
==============

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import cv2
from colour import RGB_to_XYZ, XYZ_to_RGB, RGB_COLOURSPACES,\
    XYZ_to_Lab, Lab_to_LCHab, LCHab_to_Lab, Lab_to_XYZ, XYZ_to_xyY, xyY_to_XYZ
from colour.models import BT2020_COLOURSPACE
import matplotlib.pyplot as plt

# import my libraries
import transfer_functions as tf
import test_pattern_generator2 as tpg
import color_space as cs
import plot_utility as pu
import bt2446_method_c as bmc
import bt2047_gamut_mapping as bgm
import make_bt2047_luts as mbl

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def main_func():
    img_path = "./img/step_ramp_step_65.png"
    hdr_img_non_linear = bmc.read_img_and_to_float(img_path)
    hdr_img_linear = tf.eotf(hdr_img_non_linear, tf.ST2084)
    sdr_img_linear = bmc.bt2446_method_c_tonemapping(hdr_img_linear)
    # tpg.preview_image(sdr_img_linear ** (1/2.4))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    mbl.make_bt2020_to_bt709_luts()
    # main_func()
