# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def main_func(fname):
    pp = Path(fname)
    out_fname = "./img/" + pp.name

    img = tpg.img_read_as_float(fname)
    bg_color = img[4, 4]

    st_pos1_h = 727
    st_pos1_v = 77
    ed_pos1_h = st_pos1_h + 169
    ed_pos1_v = st_pos1_v + 34

    st_pos2_h = 48
    st_pos2_v = 626
    ed_pos2_h = st_pos2_h + 1833
    ed_pos2_v = st_pos2_v + 120

    img[st_pos1_v:ed_pos1_v, st_pos1_h:ed_pos1_h] = bg_color
    img[st_pos2_v:ed_pos2_v, st_pos2_h:ed_pos2_h] = bg_color

    tpg.img_wirte_float_as_16bit_int(out_fname, img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fname_list = [
        "./img/org_tp/SDR_TyTP_P3D65.png", "./img/org_tp/HDR_tyTP_P3D65.png"
    ]
    for fname in fname_list:
        main_func(fname=fname)
