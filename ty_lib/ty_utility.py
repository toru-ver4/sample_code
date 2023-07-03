
# -*- coding: utf-8 -*-
"""
utility
"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np

# import my libraries
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def add_suffix_to_filename(fname="./img/hoge.png", suffix="_with_icc"):
    """
    Parameters
    ----------
    suffix : str
        A suffix.
    fname : str
        A file name.

    Examples
    --------
    >>> add_suffix_to_filename(fname="./img/hoge.png", suffix="_with_icc")
    "img/hoge_with_icc.png"
    """
    pp = Path(fname)
    ext = pp.suffix
    parent = str(pp.parent)
    new_name = parent + "/" + pp.stem + suffix + ext

    return new_name


def change_fname_extension(fname="./img/hoge.png", ext=".tif"):
    """
    Example
    -------
    >>> change_fname_extension(fname="./img/hoge.png", ext=".tif")
    ./img/hoge.tif
    """
    pp = Path(fname)
    parent = str(pp.parent)
    new_name = parent + "/" + pp.stem + ext

    return new_name


def conv_Nbit_to_linear(x, bit_depth=8, tf_str=tf.GAMMA24):
    """
    Examples
    --------
    >>> x = [255, 128, 64]
    >>> bit_depth = 8
    >>> tf_str = tf.ST2084
    >>> y = conv_8bit_to_linear(x, bit_depth, tf_str)
    >>> x2 = conv_linear_to_8bit(y, bit_depth, tf_str)
    >>> print(y)
    [  1.00000000e+02   9.40745992e-01   5.22570128e-02]
    >>> print(x2)
    [255 128  64]
    """
    max_value = (2 ** bit_depth) - 1
    y = tf.eotf_to_luminance(np.array(x)/max_value, tf_str)\
        / tf.REF_WHITE_LUMINANCE
    return y


def conv_linar_to_Nbit(x, bit_depth=8, tf_str=tf.GAMMA24):
    """
    Examples
    --------
    >>> x = [255, 128, 64]
    >>> bit_depth = 8
    >>> tf_str = tf.ST2084
    >>> y = conv_8bit_to_linear(x, bit_depth, tf_str)
    >>> x2 = conv_linear_to_8bit(y, bit_depth, tf_str)
    >>> print(y)
    [  1.00000000e+02   9.40745992e-01   5.22570128e-02]
    >>> print(x2)
    [255 128  64]
    """
    max_value = (2 ** bit_depth) - 1
    y = tf.oetf_from_luminance(x * tf.REF_WHITE_LUMINANCE, tf_str)
    y = np.round(y * max_value)

    if bit_depth <= 8:
        y_Nbit = np.uint8(y)
    elif bit_depth <= 16:
        y_Nbit = np.uint16(y)
    elif bit_depth <= 32:
        y_Nbit = np.uint32(y)
    else:
        y_Nbit = np.uint64(y)

    return y_Nbit


def conv_8bit_to_linear(x, tf_str=tf.GAMMA24):
    """
    Examples
    --------
    >>> x = [255, 128, 64]
    >>> tf_str = tf.ST2084
    >>> y = conv_8bit_to_linear(x, tf_str)
    >>> x2 = conv_linear_to_8bit(y, tf_str)
    >>> print(y)
    [  1.00000000e+02   9.40745992e-01   5.22570128e-02]
    >>> print(x2)
    [255 128  64]
    """
    y = conv_Nbit_to_linear(x, bit_depth=8, tf_str=tf_str)
    return y


def conv_linear_to_8bit(x, tf_str=tf.GAMMA24):
    """
    Examples
    --------
    >>> x = [255, 128, 64]
    >>> tf_str = tf.ST2084
    >>> y = conv_8bit_to_linear(x, tf_str)
    >>> x2 = conv_linear_to_8bit(y, tf_str)
    >>> print(y)
    [  1.00000000e+02   9.40745992e-01   5.22570128e-02]
    >>> print(x2)
    [255 128  64]
    """
    y = conv_linar_to_Nbit(x, bit_depth=8, tf_str=tf_str)

    return y


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fff = add_suffix_to_filename(suffix="_with_icc", fname="./img/hoge.png")
    print(fff)
