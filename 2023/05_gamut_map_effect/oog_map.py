# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
from colour.utilities import tstack
import numpy as np
from scipy import interpolate

# import my libraries
# import plot_utility as pu
import test_pattern_generator2 as tpg
import color_space as cs
import turbo_colormap
from colormap_sample import make_color_map_lut_name

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def debug_calc_delta_rgb_p():
    rgb = np.array(
        [[[0.1, 0.5, 0.7], [-0.3, 0.6, 0.1], [-0.2, 1.1, 0.5]],
         [[0.2, 0.6, 0.7], [-1.0, 1.2, 0.1], [-0.2, 1.1, -0.5]]])
    calc_delta_rgb_p(rgb=rgb)


def debug_delta_rgb_p_using_hc_pattern():
    i_fname = "./debug/BT2020-BT709_HC_Pattern.png"
    img_non_linear = tpg.img_read_as_float(i_fname)

    # img_non_linear = np.array(
    #     [[[927, 416, 221], [904, 425, 242], [882, 432, 259]]]) / 1023
    img_2020 = img_non_linear ** 2.4
    large_xyz = cs.rgb_to_large_xyz(img_2020, color_space_name=cs.BT2020)
    img_709 = cs.large_xyz_to_rgb(large_xyz, cs.BT709)
    img_709_gray = img_709[..., 0] * 0.2126 + img_709[..., 1] * 0.7152\
        + img_709[..., 2] * 0.0722
    img_709_gray = img_709_gray ** (1/2.4)
    img_709_gray = tstack([img_709_gray, img_709_gray, img_709_gray])
    delta = calc_delta_rgb_p(rgb=img_709)
    delta_turbo_img = apply_turbo_colormap(delta)
    # print(img_709[874, 256, :])
    gray_idx = delta <= 0

    out_img = np.zeros_like(img_non_linear)

    out_img[gray_idx] = img_709_gray[gray_idx]
    out_img[~gray_idx] = delta_turbo_img[~gray_idx]

    tpg.img_wirte_float_as_16bit_int(
        "./debug/delta_rgb_p.png", np.clip(out_img, 0.0, 1.0))


def debug_turbo():
    i_fname = "./debug/turbo_tp.png"
    img = tpg.img_read_as_float(i_fname) ** 2.4
    img_y = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    turbo_img = apply_turbo_colormap(img_y ** (1/2.4))
    tpg.img_wirte_float_as_16bit_int("./debug/turbo_test.png", turbo_img)


def debug_apply_oklab_colormap():
    i_fname = "./debug/map_tp.png"
    img = tpg.img_read_as_float(i_fname) ** 2.4
    img_y = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    img_y = img_y ** (1/2.4)

    color_map_lut_name = make_color_map_lut_name()
    luts = np.load(color_map_lut_name)

    for idx, lut in enumerate(luts):
        out_img = apply_colormap_from_3_1dlut(img_y, lut)
        tpg.img_wirte_float_as_16bit_int(f"./debug/cm_{idx}.png", out_img)


def load_turbo_colormap():
    """
    Examples
    --------
    >>> get_turbo_colormap()
    >>> [[ 0.18995  0.07176  0.23217]
    >>>  [ 0.19483  0.08339  0.26149]
    >>>  ...
    >>>  [ 0.49321  0.01963  0.00955]
    >>>  [ 0.4796   0.01583  0.01055]]
    """
    return np.array(turbo_colormap.turbo_colormap_data)


def apply_colormap_from_3_1dlut(x, lut):
    """
    Parameters
    ----------
    x : array_like
        input data. the data range is 0.0 -- 1.0.
    lut : array_like
        LUT data (shape is (N, 3))
    """

    # make function for linear interpolation for R, G and B.
    ref = np.linspace(0, 1, lut.shape[0])
    func_rgb = [interpolate.interp1d(ref, lut[:, idx]) for idx in range(3)]

    # apply linear interpolation
    out_rgb = [func(x) for func in func_rgb]

    return tstack(out_rgb)


def apply_turbo_colormap(x):
    """
    Parameters
    ----------
    x : array_like
        input data. the data range is 0.0 -- 1.0.
    """
    turbo = load_turbo_colormap()

    # make function for linear interpolation for R, G and B.
    ref = np.linspace(0, 1, turbo.shape[0])
    func_rgb = [interpolate.interp1d(ref, turbo[:, idx]) for idx in range(3)]

    # apply linear interpolation
    out_rgb = [func(x) for func in func_rgb]

    return tstack(out_rgb)


def calc_delta_rgb_p(rgb):
    """
    Calculate Delta R'G'B'.
    It's an indicator I developed to show the degree of Out-of-Gamut.

    Parameters
    ----------
    rgb: array_like
        A linear rgb data.

    Returns
    -------
    array_like
        Delta R'G'B'

    Examples
    --------
    >>> rgb = np.array(
    ...     [[1, 2, 3], [-3, 1, 2], [-0.1, -0.3, 2]])
    >>> calc_delta_rgb_p(rgb=rgb)
    [ 0.          1.58052192  0.71654969]
    """
    gamma = 2.4
    rgb_over_range = np.zeros_like(rgb)
    under_idx = rgb < 0
    over_idx = rgb > 1
    rgb_over_range[under_idx] = rgb[under_idx] * -1
    rgb_over_range[over_idx] = rgb[over_idx] - 1

    # print(rgb_over_range)
    rgb_p = rgb_over_range ** (1/gamma)  # "_p" means prime
    # print(rgb_p)
    d_rgb_p = np.sqrt(
        (rgb_p[..., 0] ** 2) + (rgb_p[..., 1] ** 2) + (rgb_p[..., 2] ** 2))
    # print(d_rgb_p)

    return d_rgb_p


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_calc_delta_rgb_p()
    # debug_turbo()
    debug_apply_oklab_colormap()
    # debug_delta_rgb_p_using_hc_pattern()
