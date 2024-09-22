# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path
import subprocess

# import third-party libraries
import numpy as np
import test_pattern_generator2 as tpg
import transfer_functions as tf
import color_space as cs

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_avif_specific_luminance_small_patch(
        luminance=203, size=256, cll=10000, pall=10000):
    target_cv = tf.oetf_from_luminance(luminance, tf.ST2084)
    img = np.ones((size, size, 3)) * target_cv

    png_fname = f"./debug/src_tp/patch_{luminance:05d}-nits_{size}px.png"

    print(png_fname)
    tpg.img_wirte_float_as_16bit_int(png_fname, img)

    pp = Path(png_fname)
    parent = str(pp.parent)
    if (cll is None) or (pall is None):
        avif_fname =\
            "./" + parent + "/" + pp.stem + "_cll-auto" + ".avif"
    else:
        avif_fname =\
            "./" + parent + "/" + pp.stem + f"_cll-{cll}-{pall}-nits" + ".avif"

    tpg.png_to_avif(
        png_fname=png_fname,
        avif_fname=avif_fname,
        bit_depth=12,
        color_space_name=cs.BT2020,
        transfer_characteristics=tf.ST2084,
        cll=cll,
        pall=pall,
    )


def create_specific_luminance_small_patch_all(cll_luminance=None):
    lumiannce_list = [
        60, 80, 100, 120, 140, 160, 180,
        200, 203, 204, 220, 240, 260, 280, 300, 320,
        1000, 10000]
    patch_size = 64
    if cll_luminance is None:
        cll = None
        pall = None
    else:
        cll = cll_luminance
        pall = cll_luminance
    for lumiannce in lumiannce_list:
        create_avif_specific_luminance_small_patch(
            luminance=lumiannce, size=patch_size,
            cll=cll, pall=pall
        )


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
