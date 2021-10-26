# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os
from colour.models import rgb
from colour.utilities import tstack
from colour import LCHab_to_Lab

# import third-party libraries
import numpy as np

# import my libraries
import color_space as cs
import transfer_functions as tf
from create_gamut_booundary_lut import TyLchLut,\
    make_cielab_gb_lut_fname_method_c, make_jzazbz_gb_lut_fname_method_c
from jzazbz import jzczhz_to_jzazbz
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_rgb_cv_each_hue(
        hue, cusp, focal_point_l, chroma_array, color_space_name, luminance):
    """
    Parameters
    ----------
    hue : float
        hue angle (0.0-360)
    focal_point_l : float
        lightness value of the focal_point.
    chroma_array : ndarray
        chroma value array (vertical direction of the test pattern)
    color_space_name : str
        name
    luminance : float
        luminance (ex. 1000)
    """
    hue_array_this_h_idx = np.ones_like(chroma_array) * hue
    cusp_l = cusp[0]
    cusp_c = cusp[1]
    aa = (cusp_l - focal_point_l) / cusp_c
    # bb = focal_point_l
    bb = cusp_l
    lightness_array = aa * chroma_array + bb

    ng_idx = (chroma_array > cusp_c)
    chroma_array[ng_idx] = 0.0
    lightness_array[ng_idx] = 0.0

    lch_array = tstack(
        [lightness_array, chroma_array, hue_array_this_h_idx])

    rgb_array = cs.jzazbz_to_rgb(
        jzazbz=jzczhz_to_jzazbz(lch_array),
        color_space_name=color_space_name, luminance=luminance)
    rgb_array_lumiannce = rgb_array * luminance
    if luminance > 100:
        max_luminance = 10000
        tf_str = tf.ST2084
    else:
        max_luminance = 100
        tf_str = tf.GAMMA24

    rgb_cv = tf.oetf_from_luminance(
        np.clip(rgb_array_lumiannce, 0.0, max_luminance), tf_str)

    # print(f"hue={hue}")
    # for c_idx in range(len(chroma_array)):
    #     print(f"{lch_array[c_idx]}, {rgb_cv[c_idx]}")

    return rgb_cv


def create_cl_pattern():
    color_space_name = cs.BT709
    width = 1920
    height = 1080
    hue_num = 42
    hue_array = np.linspace(0, 360, hue_num, endpoint=False)
    chroma_num = int(round(height / width * hue_num))
    luminance = 100

    # load lut
    lut_name = make_jzazbz_gb_lut_fname_method_c(
        color_space_name=color_space_name, luminance=luminance)
    lut = TyLchLut(np.load(lut_name))
    focal_point_l = lut.ll_max / 2

    # find maximum chroma from the cups of all hue angles
    cusp = np.zeros((hue_num, 3))
    for h_idx in range(hue_num):
        hue = hue_array[h_idx]
        # cusp[h_idx] = lut.get_cusp(hue=hue)
        cusp[h_idx] = lut.get_cusp_without_intp(hue=hue)
    max_idx = np.argmax(cusp[..., 1])
    max_cusp = cusp[max_idx]
    chroma_max = max_cusp[1]
    lightness_max = max_cusp[0]
    print(max_cusp)
    # chroma_max = np.max(cusp[..., 1])
    # print(chroma_max)

    # # calc each delta_Ez
    delta_Ez_total\
        = (chroma_max ** 2 + (focal_point_l - lightness_max) ** 2) ** 0.5
    delta_Ez = delta_Ez_total / (chroma_num - 1)

    block_width_list = tpg.equal_devision(width, hue_num)
    block_height_list = tpg.equal_devision(height, chroma_num)

    # calc each pixel value
    chroma_array = np.linspace(0, chroma_max, chroma_num)
    print(chroma_array)
    h_buf = []
    for h_idx in range(hue_num):
        hue = hue_array[h_idx]
        rgb_cv = calc_rgb_cv_each_hue(
            hue=hue, cusp=cusp[h_idx], focal_point_l=focal_point_l,
            chroma_array=chroma_array.copy(),
            color_space_name=color_space_name, luminance=luminance)
        block_width = block_width_list[h_idx]

        v_buf = []
        for c_idx in range(chroma_num):
            block_height = block_height_list[c_idx]
            block_img = np.ones((block_height, block_width, 3)) * rgb_cv[c_idx]
            v_buf.append(block_img)
        h_buf.append(np.vstack(v_buf))
    img = np.hstack(h_buf)

    tpg.img_wirte_float_as_16bit_int(f"./img/abb-{luminance}.png", img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_cl_pattern()
