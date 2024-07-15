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

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def encode_HDR_TP():
    src_file_list = [
        "./debug/src_tp/SMPTE ST2084_ITU-R BT.2020_D65_1920x1080_rev07_type1.png",
        "./debug/src_tp/SMPTE ST2084_ITU-R BT.2020_D65_3840x2160_rev07_type1.png",
        "./debug/src_tp/SMPTE ST2084_P3-D65_D65_1920x1080_rev07_type1.png",
        "./debug/src_tp/SMPTE ST2084_P3-D65_D65_3840x2160_rev07_type1.png",
    ]

    for png_fname in src_file_list:
        print(png_fname)
        if png_fname.find("BT.2020") > -1:
            cicp = "9/16/0"
        elif png_fname.find("P3-D65") > -1:
            cicp = "12/16/0"
        else:
            raise ValueError("filename dosn't contain colorimetry infomation.")

        pp = Path(png_fname)
        parent = str(pp.parent)
        avif_fname = "./" + parent + "/" + pp.stem + ".avif"
        cmd = [
            "avifenc", png_fname, "-d", "10", "-y", "444", "--cicp", cicp,
            "-r", "full", "--lossless", "--ignore-exif", avif_fname
        ]
        print(" ".join(cmd))
        subprocess.run(cmd)


def create_specific_luminance_small_patch(luminance=203, size=256):
    target_cv = tf.oetf_from_luminance(luminance, tf.ST2084)
    img = np.ones((size, size, 3)) * target_cv

    png_fname = f"./debug/src_tp/patch_{luminance:05d}-nits_{size}px.png"

    print(png_fname)
    tpg.img_wirte_float_as_16bit_int(png_fname, img)

    pp = Path(png_fname)
    parent = str(pp.parent)
    avif_fname = "./" + parent + "/" + pp.stem + ".avif"
    cmd = [
        "avifenc", png_fname, "-d", "10", "-y", "444", "--cicp", "9/16/9",
        "-r", "full", "--lossless", "--ignore-exif", avif_fname
    ]
    print(" ".join(cmd))
    subprocess.run(cmd)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_specific_luminance_small_patch(luminance=1000, size=512)
    encode_HDR_TP()
