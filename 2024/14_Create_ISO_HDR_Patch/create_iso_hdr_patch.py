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


def create_png_patch(
        luminance: int=203,
        size: int=256,
        png_fname: str="out.png"
) -> str:
    target_cv = tf.oetf_from_luminance(luminance, tf.ST2084)
    img = np.ones((size, size, 3)) * target_cv

    print(png_fname)
    tpg.img_wirte_float_as_16bit_int(png_fname, img)

    return png_fname


def make_avif_name_from_png_fname(
        cll: int=10000,
        pall: int=10000,
        png_fname: str="./img/out.png"
) -> str:
    pp = Path(png_fname)
    parent = str(pp.parent).replace("png", "avif")
    if (cll is None) or (pall is None):
        avif_fname =\
            "./" + parent + "/" + pp.stem + "_cll-auto" + ".avif"
    else:
        avif_fname =\
            "./" + parent + "/" + pp.stem + f"_cll-{cll}-{pall}-nits" + ".avif"

    return avif_fname


def make_jxl_name_from_png_fname(
        png_fname: str="./img/out.png"
) -> str:
    pp = Path(png_fname)
    parent = str(pp.parent).replace("png", "jxl")
    jxl_fname =\
            "./" + parent + "/" + pp.stem + ".jxl"

    return jxl_fname


def create_avif_from_png_file(
        cll: int=10000,
        pall: int=10000,
        png_fname: str="./img/out.png",
        avif_fname: str="./img/out.avif"
):

    tpg.png_to_avif(
        png_fname=png_fname,
        avif_fname=avif_fname,
        bit_depth=12,
        color_space_name=cs.BT2020,
        transfer_characteristics=tf.ST2084,
        cll=cll,
        pall=pall,
    )


def create_jxl_from_png_file(
        png_fname: str="./img/out.png",
        jxl_fname: str="./img/out.jxl"
):
    tpg.png_to_jxl(
        png_fname=png_fname,
        jxl_fname=jxl_fname,
        bit_depth=10,
        white_point="D65",
        color_space_name=cs.BT2020,
        transfer_characteristics=tf.ST2084
    )


def main_func():
    # parameters
    luminance_list = [
        60, 80, 100, 120, 140, 160, 180,
        200, 203, 204, 220, 240, 260, 280, 300, 320,
        1000, 10000]
    patch_size = 64

    png_fname_list = []
    for luminance in luminance_list:
        # create src png files
        png_fname\
            = f"./img/png/patch_{luminance:05d}-nits_{patch_size}px.png"
        create_png_patch(
            luminance=luminance, size=patch_size, png_fname=png_fname)

        # create avif patch
        cll_param_list = [10000, None]
        for cll_param in cll_param_list:
            avif_fname = make_avif_name_from_png_fname(
                cll=cll_param, pall=cll_param, png_fname=png_fname
            )
            print(avif_fname)
            create_avif_from_png_file(
                cll=cll_param, pall=cll_param,
                png_fname=png_fname, avif_fname=avif_fname
            )

        # create jxl patch
        jxl_fname = make_jxl_name_from_png_fname(png_fname=png_fname)
        create_jxl_from_png_file(png_fname=png_fname, jxl_fname=jxl_fname)
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
