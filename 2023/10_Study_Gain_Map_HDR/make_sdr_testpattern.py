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
from ty_utility import add_suffix_to_filename

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def remove_hdr_info_text(fname):
    pp = Path(fname)
    out_fname = "./img/" + pp.name

    img = tpg.img_read_as_float(fname)
    bg_color = img[4, 4]

    st_pos1_h = 727
    st_pos1_v = 77
    ed_pos1_h = st_pos1_h + 169
    ed_pos1_v = st_pos1_v + 34

    # st_pos2_h = 48
    # st_pos2_v = 626
    # ed_pos2_h = st_pos2_h + 1833
    # ed_pos2_v = st_pos2_v + 120

    img[st_pos1_v:ed_pos1_v, st_pos1_h:ed_pos1_h] = bg_color
    # img[st_pos2_v:ed_pos2_v, st_pos2_h:ed_pos2_h] = bg_color

    print(out_fname)
    tpg.img_wirte_float_as_16bit_int(out_fname, img)
    return out_fname


def make_sdr_version_tp(hdr_fname):
    sdr_fname = hdr_tp_fname.replace("HDR", "SDR")
    img = tpg.img_read_as_float(hdr_tp_fname)
    linear = tf.eotf_to_luminance(img, tf.ST2084)
    ref_white = 203
    linear[linear > ref_white] = ref_white

    sdr_img = tf.oetf(linear / ref_white, tf.GAMMA24)
    print(sdr_fname)
    tpg.img_wirte_float_as_16bit_int(sdr_fname, sdr_img)


def make_reduced_luminance_hdr_tp(hdr_tp_fname, target_luminance=10000):
    img = tpg.img_read_as_float(hdr_tp_fname)
    linear = tf.eotf_to_luminance(img, tf.ST2084)
    linear[linear > target_luminance] = target_luminance
    img_out = tf.oetf_from_luminance(linear, tf.ST2084)

    out_fname = add_suffix_to_filename(
        fname=hdr_tp_fname, suffix=f"_{target_luminance}")
    print(out_fname)
    tpg.img_wirte_float_as_16bit_int(out_fname, img_out)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    hdr_tp_fname = remove_hdr_info_text(
        fname="./img/org_tp/HDR_tyTP_P3D65.png")
    target_luminance_list = [
        600, 1000, 4000, 10000
    ]
    for target_luminance in target_luminance_list:
        make_reduced_luminance_hdr_tp(
            hdr_tp_fname=hdr_tp_fname, target_luminance=target_luminance)
    # make_sdr_version_tp(hdr_fname=hdr_tp_fname)
