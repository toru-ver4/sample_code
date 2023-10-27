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


GAIN_MAP_CS_NAME = cs.P3_D65
K_SDR = 10 ** -5
K_HDR = 10 ** -5
GAIN_MAP_ENCODE_GAMMA = 1.0


def apply_gain_map_to_sdr():
    pass


def linearize_input_image(fname, tf_name=tf.ST2084, cs_name=cs.P3_D65):
    img = tpg.img_read_as_float(fname)
    linear_img = tf.eotf_to_luminance(img, tf_name) / tf.REF_WHITE_LUMINANCE
    large_xyz = cs.rgb_to_large_xyz(
        rgb=linear_img, color_space_name=cs_name)
    linear_img = cs.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=GAIN_MAP_CS_NAME)

    return linear_img


def create_gain_map_fname(sdr_fname, hdr_fname):
    """
    Examples
    --------
    >>> sdr_fname = "./img/SDR_TyTP_P3D65.png"
    >>> hdr_fname = "./img/HDR_tyTP_P3D65.png"
    >>> gain_map_fname = create_gain_map_fname(sdr_fname, hdr_fname)
    >>> print(gain_map_fname)
    ./gain_map/gain_map_SDR_TyTP_P3D65_HDR_tyTP_P3D65.png
    """
    pp_sdr = Path(sdr_fname)
    pp_hdr = Path(hdr_fname)

    base_dir = "./gain_map"
    sdr_name = pp_sdr.stem
    hdr_name = pp_hdr.stem

    gain_map_fname = f"{str(base_dir)}/gain_map_{sdr_name}_{hdr_name}.png"

    return gain_map_fname


def create_gain_map_metadata_fname(gain_map_fname):
    return f"{gain_map_fname}_meta.npy"


def create_log2_gain_map_image(sdr_linear, hdr_linear):
    """
    Parameters
    ----------
    sdr_linear : ndarray
        SDR image
    hdr_linear : ndarray
        HDR image
    Note
    ----
    SDR image and HDR image must be the same color gamut.
    """
    def calc_min_val_color(gain_map):
        min_val = [np.min(gain_map[..., idx]) for idx in range(3)]
        return np.array(min_val)

    def calc_max_val_color(gain_map):
        max_val = [np.max(gain_map[..., idx]) for idx in range(3)]
        return np.array(max_val)
    gain_map = np.log2((hdr_linear + K_HDR)/(sdr_linear + K_SDR))
    min_val = calc_min_val_color(gain_map=gain_map)
    max_val = calc_max_val_color(gain_map=gain_map)

    gain_map = (gain_map - min_val) / (max_val - min_val)

    return gain_map, min_val, max_val


def save_gain_map_img(
        gain_map_img_log2, min_val, max_val, gain_map_fname):
    gain_map_metadata_fname = create_gain_map_metadata_fname(
        gain_map_fname=gain_map_fname)

    # apply oetf
    gain_map_img_log2 = gain_map_img_log2 ** (1/GAIN_MAP_ENCODE_GAMMA)

    # save
    pass


def create_and_save_gain_map(
        sdr_fname="./img/SDR_TyTP_P3D65.png",
        hdr_fname="./img/HDR_tyTP_P3D65.png",
        sdr_cs_name=cs.P3_D65, hdr_cs_name=cs.P3_D65,
        sdr_tf_name=tf.GAMMA24, hdr_tf_name=tf.ST2084):

    # define
    gain_map_fname = create_gain_map_fname(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname)

    # linearize
    sdr_liner = linearize_input_image(
        fname=sdr_fname, tf_name=sdr_tf_name, cs_name=sdr_cs_name)
    hdr_liner = linearize_input_image(
        fname=hdr_fname, tf_name=hdr_tf_name, cs_name=hdr_cs_name)

    # create gain map
    gain_map_img_log2, min_val, max_val = create_log2_gain_map_image(
        sdr_linear=sdr_liner, hdr_linear=hdr_liner)

    # save gain map
    save_gain_map_img(
        gain_map_img_log2=gain_map_img_log2,
        min_val=min_val, max_val=max_val, gain_map_fname=gain_map_fname)


def create_intermediate_hdr_image():
    pass


def apply_gain_map(
        sdr_fname="./img/SDR_TyTP_P3D65.png",
        gain_map_img="./img/gain_map_SDR_TyTP_P3D65_HDR_TyTP_P3D65.png",
        sdr_white_nit=203, hdr_white_nit=10000):
    """
    2種類の出力をするかもしれない。
    1つは正直ベースにパラメータをファイル名に埋め込んだ版。
    もう1つは YouTube アップロード用にパラメータは消して連番ファイルにした版。
    なお、パラメータは文字列として画像に焼く
    """
    pass


def debug_func():
    sdr_fname = "./img/SDR_TyTP_P3D65.png"
    hdr_fname = "./img/HDR_tyTP_P3D65.png"
    sdr_cs_name = cs.P3_D65
    hdr_cs_name = cs.P3_D65
    sdr_tf_name = tf.GAMMA24
    hdr_tf_name = tf.ST2084
    create_and_save_gain_map(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname,
        sdr_cs_name=sdr_cs_name, hdr_cs_name=hdr_cs_name,
        sdr_tf_name=sdr_tf_name, hdr_tf_name=hdr_tf_name)

    sdr_white_nit = 203
    hdr_white_nit = 1000
    gain_map_img_name = create_gain_map_fname(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname)
    apply_gain_map(
        sdr_fname=sdr_fname, gain_map_img=gain_map_img_name,
        sdr_white_nit=sdr_white_nit, hdr_white_nit=hdr_white_nit)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
