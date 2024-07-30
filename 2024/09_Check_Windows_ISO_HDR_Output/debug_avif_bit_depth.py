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


def create_specific_bit_depth_avif():
    wrgbmyc = tpg.create_wrgbmyc_ramp_data(bit_depth=10)
    max_cv = (2 ** 10) - 1
    img = wrgbmyc / max_cv

    png_fname = "./debug/src_tp/debug_avif_precision.png"
    tpg.img_wirte_float_as_16bit_int(png_fname, img)

    bit_depth_list = [10, 12]

    for bit_depth in bit_depth_list:
        avif_fname\
            = f"./debug/src_tp/debug_avif_precision_{bit_depth}-bit.avif"
        cmd = [
            "avifenc", png_fname,
            "-d", f"{bit_depth}",
            "-y", "444",
            "--cicp", "9/16/0",
            "-r", "full",
            "--lossless",
            "--ignore-exif",
            avif_fname
        ]
        print(" ".join(cmd))
        subprocess.run(cmd)


def decode_specific_bit_depth_avif():
    bit_depth_list = [10, 12]

    for bit_depth in bit_depth_list:
        avif_fname\
            = f"./debug/src_tp/debug_avif_precision_{bit_depth}-bit.avif"
        png_fname\
            = f"./debug/src_tp/debug_avif_precision_{bit_depth}-bit_decoded.png"
        cmd = [
            "avifdec",
            "-d", "16",
            avif_fname,
            png_fname
        ]
        print(" ".join(cmd))
        subprocess.run(cmd)


def verify_data():
    ref_wrgbmyc = tpg.create_wrgbmyc_ramp_data(bit_depth=10)
    fname = "./debug/src_tp/debug_avif_precision_10-bit_decoded.png"
    img = np.round(tpg.img_read_as_float(fname) * 1023)\
        .astype(np.uint16)
    diff = img.astype(np.int16) - ref_wrgbmyc.astype(np.int16)
    print(np.sum(diff))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_specific_bit_depth_avif()
    # decode_specific_bit_depth_avif()
    # verify_data()
    print(tf.eotf_to_luminance(900/1023, tf.ST2084))
