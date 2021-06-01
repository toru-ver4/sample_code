# -*- coding: utf-8 -*-
"""
plot gamut boundary
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_valid_ab_plane_image_srgb(
        l_val=50, ab_max=200, ab_sample=512, color_space_name=cs.BT2020,
        bg_rgb=np.array([0.5, 0.5, 0.5])):
    """
    Create an image that indicates the valid area of the ab plane.

    Parameters
    ----------
    l_val : float
        A Lightness value. range is 0.0 - 100.0
    ab_max : float
        A maximum value of the a, b range.
    ab_sapmle : int
        A number of samples in the image resolution.
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    """
    dummy_rgb = bg_rgb
    aa_base = np.linspace(-ab_max, ab_max, ab_sample)
    bb_base = np.linspace(-ab_max, ab_max, ab_sample)
    aa = aa_base.reshape((1, ab_sample))\
        * np.ones_like(bb_base).reshape((ab_sample, 1))
    bb = bb_base.reshape((ab_sample, 1))\
        * np.ones_like(aa_base).reshape((1, ab_sample))
    ll = np.ones_like(aa) * l_val
    lab = np.dstack((ll, aa, bb[::-1])).reshape((ab_sample, ab_sample, 3))
    rgb = cs.lab_to_rgb(lab=lab, color_space_name=color_space_name)
    ng_idx = cgb.is_out_of_gamut_rgb(rgb=rgb)
    rgb[ng_idx] = dummy_rgb
    srgb_image = tf.oetf(np.clip(rgb, 0.0, 1.0), tf.SRGB)

    return srgb_image


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
