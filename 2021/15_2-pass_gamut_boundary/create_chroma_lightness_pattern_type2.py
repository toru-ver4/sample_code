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

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_cl_pattern():
    color_space_name = cs.BT709
    width = 1920
    height = 1080
    hue_num = 24
    hue_array = np.linspace(0, 360, hue_num, endpoint=False)
    chroma_num = int(round(height / width * hue_num))
    luminance = 1000

    # load lut
    lut_name = make_jzazbz_gb_lut_fname_method_c(
        color_space_name=color_space_name, luminance=luminance)
    lut = TyLchLut(np.load(lut_name))
    focal_point_l = lut.ll_max / 2

    # find maximum chroma from the cups of all hue angles
    cusp = np.zeros((hue_num, 3))
    for h_idx in range(hue_num):
        hue = hue_array[h_idx]
        cusp[h_idx] = lut.get_cusp(hue=hue)
    chroma_max = np.max(cusp[..., 1])
    print(chroma_max)

    # calc each pixel value
    chroma_array = np.linspace(0, chroma_max, chroma_num)
    print(chroma_array)
    for h_idx in range(hue_num):
        hue = hue_array[h_idx]
        hue_array_this_h_idx = np.ones_like(chroma_array) * hue
        cusp_l = cusp[h_idx][0]
        cusp_c = cusp[h_idx][1]
        aa = (cusp_l - focal_point_l) / cusp_c
        bb = focal_point_l
        lightness_array = aa * chroma_array + bb
        lch_array = tstack(
            [lightness_array, chroma_array, hue_array_this_h_idx])
        rgb_array = cs.jzazbz_to_rgb(
            jzazbz=jzczhz_to_jzazbz(lch_array),
            color_space_name=color_space_name, luminance=luminance)
        rgb_array_lumiannce = rgb_array * luminance
        print(f"h_idx={h_idx}")
        print(f"cusp={cusp[h_idx]}")
        for c_idx in range(chroma_num):
            print(f"{lch_array[c_idx]}, {rgb_array_lumiannce[c_idx]}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_cl_pattern()
