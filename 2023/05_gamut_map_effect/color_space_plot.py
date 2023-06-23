# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import XYZ_to_RGB, LCHab_to_Lab
from colour.models import RGB_COLOURSPACES

# import my libraries
import color_space as cs
import transfer_functions as tf
from create_gamut_booundary_lut import is_out_of_gamut_rgb

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_valid_oklab_ab_plane_image_gm24(
        l_val=50, ab_max=100, ab_sample=512, color_space_name=cs.BT2020,
        bg_rgb_luminance=np.array([0.5, 0.5, 0.5])):
    """
    Create an image that indicates the valid area of the ab plane.

    Parameters
    ----------
    j_val : float
        A Lightness value. range is 0.0 - 1.0
    ab_max : float
        A maximum value of the a, b range.
    ab_sapmle : int
        A number of samples in the image resolution.
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    maximum_luminance : float
        maximum luminance of the target display device.
    """
    aa_base = np.linspace(-ab_max, ab_max, ab_sample)
    bb_base = np.linspace(-ab_max, ab_max, ab_sample)
    aa = aa_base.reshape(-1, ab_sample).repeat(ab_sample, axis=0)
    bb = bb_base.reshape(ab_sample, -1).repeat(ab_sample, axis=1)

    ll = np.ones_like(aa) * l_val
    lab = np.dstack((ll, aa, bb[::-1])).reshape((ab_sample, ab_sample, 3))
    rgb = cs.oklab_to_rgb(oklab=lab, color_space_name=color_space_name)

    ng_idx = is_out_of_gamut_rgb(rgb=rgb)
    rgb[ng_idx] = bg_rgb_luminance
    rgb_gm24 = tf.oetf(np.clip(rgb, 0.0, 1.0), tf.GAMMA24)

    return rgb_gm24


def create_valid_oklab_cl_plane_image_gm24(
        h_val=50, c_max=220, c_sample=1280, l_sample=720,
        color_space_name=cs.BT2020, bg_val=0.5):
    """
    Create an image that indicates the valid area of the ab plane.

    Parameters
    ----------
    h_val : float
        A Hue value. range is 0.0 - 360.0
    c_max : float
        A maximum value of the chroma.
    c_sapmle : int
        A number of samples for the chroma.
    l_sample : int
        A number of samples for the lightness.
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    bg_lightness : float
        background lightness value.
    """
    l_max = 1.0
    dummy_rgb = np.array([bg_val, bg_val, bg_val])

    cc_base = np.linspace(0, c_max, c_sample)
    ll_base = np.linspace(0, l_max, l_sample)
    cc = cc_base.reshape(1, c_sample)\
        * np.ones_like(ll_base).reshape(l_sample, 1)
    ll = ll_base.reshape(l_sample, 1)\
        * np.ones_like(cc_base).reshape(1, c_sample)
    hh = np.ones_like(cc) * h_val

    lch = np.dstack([ll[::-1], cc, hh]).reshape((l_sample, c_sample, 3))
    lab = LCHab_to_Lab(lch)
    rgb = cs.oklab_to_rgb(oklab=lab, color_space_name=color_space_name)
    ng_idx = is_out_of_gamut_rgb(rgb=rgb)
    # ng_idx = cgb.is_outer_gamut(lab=lab, color_space_name=color_space_name)
    rgb[ng_idx] = dummy_rgb

    rgb_gm24 = tf.oetf(np.clip(rgb, 0.0, 1.0), tf.GAMMA24)

    return rgb_gm24


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
