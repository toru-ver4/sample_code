# -*- coding: utf-8 -*-
"""
debug
==============

"""

# import standard libraries
import os
import ctypes
import platform

# import third-party libraries
import numpy as np
import cv2
from colour import RGB_to_XYZ, XYZ_to_RGB, RGB_COLOURSPACES,\
    XYZ_to_Lab, Lab_to_LCHab, LCHab_to_Lab, Lab_to_XYZ, XYZ_to_xyY, xyY_to_XYZ
from colour import read_LUT
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


def make_dpi_aware():
    """
    https://github.com/PySimpleGUI/PySimpleGUI/issues/1179
    """
    if int(platform.release()) >= 8:
        ctypes.windll.shcore.SetProcessDpiAwareness(True)


def main_func():
    img_path = "./img/high.png"
    hdr_img_non_linear = bmc.read_img_and_to_float(img_path)
    hdr_img_linear = tf.eotf(hdr_img_non_linear, tf.ST2084)
    sdr_img_linear = bmc.bt2446_method_c_tonemapping(hdr_img_linear)

    tpg.preview_image(sdr_img_linear ** (1/2.4))

    sdr_709_liner = bgm.bt2407_gamut_mapping_for_rgb_linear(
        rgb_linear=sdr_img_linear,
        outer_color_space_name=cs.BT2020,
        inner_color_space_name=cs.BT709)
    tpg.preview_image(sdr_709_liner ** (1/2.4))


def get_youtube_tonemap_line():
    x_pq = np.linspace(0, 1, 1024)
    x_img = np.dstack([x_pq, x_pq, x_pq])
    lut3d = read_LUT("./luts/HDR10_to_BT709_YouTube_Rev03.cube")
    y = lut3d.apply(x_img)

    plt.plot(x_pq, y[..., 1].flatten())
    x = tf.eotf_to_luminance(x_pq, tf.ST2084)
    y = tf.eotf_to_luminance(y[..., 1].flatten(), tf.GAMMA24)
    out_data = np.dstack((x, y)).reshape((1024, 2))
    print(out_data)
    np.save("./youtube.npy", out_data)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # make_dpi_aware()
    # mbl.make_bt2020_to_bt709_luts()
    # main_func()
    get_youtube_tonemap_line()
