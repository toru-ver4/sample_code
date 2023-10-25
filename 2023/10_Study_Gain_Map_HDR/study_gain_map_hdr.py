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


def linearize_input_image(fname, tf_name=tf.ST2084, cs_name=cs.P3_D65):
    pass


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


def create_gain_map(
        sdr_fname, hdr_fname, sdr_tf=tf.GAMMA24, hdr_tf=tf.ST2084):
    """
    Parameters
    ----------
    Note
    ----
    SDR image and HDR image must be the same color gamut.
    """
    gain_map_fname = create_gain_map_fname(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname)
    print(gain_map_fname)


def debug_func():
    sdr_fname = "./img/SDR_TyTP_P3D65.png"
    hdr_fname = "./img/HDR_tyTP_P3D65.png"
    create_gain_map(sdr_fname=sdr_fname, hdr_fname=hdr_fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
