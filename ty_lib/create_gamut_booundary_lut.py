# -*- coding: utf-8 -*-
"""
create gamut boundary lut.
"""

# import standard libraries
import os
import sys
import ctypes
from colour.models.rgb.derivation import RGB_luminance

# import third-party libraries
import numpy as np
from colour.utilities import tstack
from colour import LCHab_to_Lab, Lab_to_XYZ, XYZ_to_RGB, xy_to_XYZ, xyY_to_XYZ
from colour import RGB_COLOURSPACES
from multiprocessing import Pool, cpu_count
# from multiprocessing import shared_memory
from multiprocessing import Array
from scipy import signal, interpolate

# import my libraries
import color_space as cs
from common import MeasureExecTime
from jzazbz import jzazbz_to_large_xyz, jzczhz_to_jzazbz, large_xyz_to_jzazbz

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

L_SAMPLE_NUM_MAX = 1024
H_SAMPLE_NUM_MAX = 4096
COLOR_NUM = 3
FLOAT_SIZE = 4
JZAZBZ_CHROMA_MAX = 0.5
CIELAB_CHROMA_MAX = 250
L_SAMPLE_DEFAULT = 1024
H_SAMPLE_DEFAULT = 4096

shared_array_type = ctypes.c_float
shared_array_elem_size = ctypes.sizeof(shared_array_type)
shared_array = Array(
    typecode_or_type=shared_array_type,
    size_or_initializer=L_SAMPLE_NUM_MAX*H_SAMPLE_NUM_MAX*COLOR_NUM)

# shm = shared_memory.SharedMemory(
#     create=True, size=H_SAMPLE_NUM_MAX*COLOR_NUM*FLOAT_SIZE)
# shm_buf = np.ndarray(
#     (1, H_SAMPLE_NUM_MAX, 3), dtype=np.float32, buffer=shm.buf)

DELTA = 10 ** -8


def is_outer_gamut(lab, color_space_name):
    min_val = -DELTA
    max_val = 1 + DELTA
    rgb = XYZ_to_RGB(
        Lab_to_XYZ(lab), cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].matrix_XYZ_to_RGB)
    r_judge = (rgb[..., 0] < min_val) | (rgb[..., 0] > max_val)
    g_judge = (rgb[..., 1] < min_val) | (rgb[..., 1] > max_val)
    b_judge = (rgb[..., 2] < min_val) | (rgb[..., 2] > max_val)
    judge = (r_judge | g_judge) | b_judge

    return judge


def is_outer_gamut_jzazbz(
        jzazbz, color_space_name, luminance=10000, delta=DELTA):
    min_val = -delta
    max_val = 1 + delta
    rgb = XYZ_to_RGB(
        jzazbz_to_large_xyz(jzazbz), cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].matrix_XYZ_to_RGB)
    rgb = rgb / luminance
    r_judge = (rgb[..., 0] < min_val) | (rgb[..., 0] > max_val)
    g_judge = (rgb[..., 1] < min_val) | (rgb[..., 1] > max_val)
    b_judge = (rgb[..., 2] < min_val) | (rgb[..., 2] > max_val)
    judge = (r_judge | g_judge) | b_judge

    return judge


def is_out_of_gamut_rgb(rgb, delta=DELTA):
    min_val = -delta
    max_val = 1 + delta
    r_judge = (rgb[..., 0] < min_val) | (rgb[..., 0] > max_val)
    g_judge = (rgb[..., 1] < min_val) | (rgb[..., 1] > max_val)
    b_judge = (rgb[..., 2] < min_val) | (rgb[..., 2] > max_val)
    judge = (r_judge | g_judge) | b_judge

    return judge


def calc_chroma_boundary_specific_l(
        ll, chroma_sample, chroma_max, hue_num, cs_name):
    """
    parameters
    ----------
    ll : float
        L* value(CIELAB)
    chroma_sample : int
        Sample number of the Chroma
    chroma_max : float
        The maximum value of the Chroma search range.
    hue_num : int
        Sample number of the Hue
    cs_name : string
        A color space name. ex. "ITU-R BT.709", "ITU-R BT.2020"
    """
    # lch --> rgb
    hue_max = 360
    hue_base = np.linspace(0, hue_max, hue_num)
    chroma_base = np.linspace(0, chroma_max, chroma_sample)
    hh = hue_base.reshape((hue_num, 1))\
        * np.ones_like(chroma_base).reshape((1, chroma_sample))
    cc = chroma_base.reshape((1, chroma_sample))\
        * np.ones_like(hue_base).reshape((hue_num, 1))
    ll = np.ones_like(hh) * ll

    lch = tstack((ll, cc, hh))
    lab = LCHab_to_Lab(lch)
    rgb = cs.lab_to_rgb(lab=lab, color_space_name=cs_name)
    ng_idx = is_out_of_gamut_rgb(rgb=rgb)
    # print(lch)
    # print(lab)
    # print(ng_idx)
    # arg_ng_idx = np.argwhere(ng_idx > 0)
    chroma_array = np.zeros(hue_num)
    for h_idx in range(hue_num):
        chroma_ng_idx_array = np.where(ng_idx[h_idx] > 0)
        chroma_ng_idx = np.min(chroma_ng_idx_array)
        chroma_ng_idx = chroma_ng_idx - 1 if chroma_ng_idx > 0 else 0
        chroma_array[h_idx] = chroma_ng_idx / (chroma_sample - 1) * chroma_max
    # print(chroma_array)

    return hue_base, chroma_array


def calc_chroma_boundary_specific_hue_method_b(
        hue, chroma_sample, lightness_sample, cs_name, **kwargs):
    """
    parameters
    ----------
    hue : float
        Hue value(CIELAB). range is 0.0 - 360.
    chroma_sample : int
        Sample number of the Chroma
    lightness_num : int
        Sample number of the Lightness
    cs_name : string
        A color space name. ex. "ITU-R BT.709", "ITU-R BT.2020"
    """
    # lch --> rgb
    ll_max = 100
    ll_base = np.linspace(0, ll_max, lightness_sample)
    chroma_max = CIELAB_CHROMA_MAX
    chroma_base = np.linspace(0, chroma_max, chroma_sample)
    ll = ll_base.reshape((lightness_sample, 1))\
        * np.ones_like(chroma_base).reshape((1, chroma_sample))
    cc = chroma_base.reshape((1, chroma_sample))\
        * np.ones_like(ll_base).reshape((lightness_sample, 1))
    hh = np.ones_like(ll) * hue

    lch = tstack((ll, cc, hh))
    lab = LCHab_to_Lab(lch)
    rgb = cs.lab_to_rgb(lab=lab, color_space_name=cs_name)
    ng_idx = is_out_of_gamut_rgb(rgb=rgb)
    # print(lch)
    # print(lab)
    # print(ng_idx)
    chroma_array = np.zeros(lightness_sample)
    for h_idx in range(lightness_sample):
        chroma_ng_idx_array = np.where(ng_idx[h_idx] > 0)
        chroma_ng_idx = np.min(chroma_ng_idx_array)
        chroma_ng_idx = chroma_ng_idx - 1 if chroma_ng_idx > 0 else 0
        chroma_array[h_idx] = chroma_ng_idx / (chroma_sample - 1) * chroma_max
    # print(chroma_array)

    return ll_base, chroma_array


def thread_wrapper_calc_chroma_boundary_method_b_specific_hue(args):
    lightness_array, chroma_array =\
        calc_chroma_boundary_specific_hue_method_b(**args)
    hue_array = np.ones_like(lightness_array) * args['hue']
    plane_lut = tstack([lightness_array, chroma_array, hue_array])
    ll_len = args['lightness_sample']
    hh_len = args['hue_sample']
    h_idx = args['hue_idx']

    hue_plane_size = hh_len * 3

    for l_idx in range(ll_len):
        addr = (hue_plane_size * l_idx) + (h_idx * 3)
        shared_array[addr:addr+3] = np.float32(plane_lut[l_idx])


def create_cielab_gamut_boundary_lut_method_b(
        lightness_sample, chroma_sample, hue_sample, cs_name):
    """
    parameters
    ----------
    lightness_sample : int
        Sample number of the Lightness
        Lightness range is 0.0 - 100.0
    chroma_sample : int
        Sample number of the Chroma
    hue_sample : int
        Sample number of the Hue
    cs_name : string
        A color space name. ex. "ITU-R BT.709", "ITU-R BT.2020"
    """

    total_process_num = hue_sample
    block_process_num = 16
    # block_process_num = 3  # for 32768 sample
    block_num = int(round(total_process_num / block_process_num + 0.5))

    mtime = MeasureExecTime()
    mtime.start()
    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, h_idx={h_idx}")  # User
            if h_idx >= total_process_num:                         # User
                break
            d = dict(
                hue=h_idx/(hue_sample-1)*360, chroma_sample=chroma_sample,
                lightness_sample=lightness_sample, cs_name=cs_name,
                hue_sample=hue_sample, hue_idx=h_idx)
            args.append(d)
            # thread_wrapper_calc_chroma_boundary_specific_hue(d)
        with Pool(cpu_count()) as pool:
            pool.map(
                thread_wrapper_calc_chroma_boundary_method_b_specific_hue,
                args)
            # mtime.lap()
        mtime.lap()
    mtime.end()

    lut = np.array(
        shared_array[:lightness_sample*hue_sample*3]).reshape(
            (lightness_sample, hue_sample, 3))

    return lut


def get_gamut_boundary_lch_from_lut(lut, lh_array, lightness_max=100):
    """
    parameters
    ----------
    lut : ndarray
        A Gamut boundary lut
    lh_array : ndarray
        lightness, hue array for interpolate.
    lightness_max : float
        normalize value for lightness

    Examples
    --------
    >>> lut = np.load("./lut/lut_sample_11_9_8192_ITU-R BT.2020.npy")
    >>> ll_base = np.linspace(0, 100, 11)
    >>> hh_base = np.ones_like(ll_base) * 20
    >>> lh_array = tstack([ll_base, hh_base])
    >>> print(lh_array)
    [[   0.   20.]
     [  10.   20.]
     [  20.   20.]
     [  30.   20.]
     [  40.   20.]
     [  50.   20.]
     [  60.   20.]
     [  70.   20.]
     [  80.   20.]
     [  90.   20.]
     [ 100.   20.]]
    >>> lch = get_gamut_boundary_lch_from_lut(lut=lut, lh_array=lh_array)
    [[   0.            0.           20.        ]
     [  10.           33.74652269   20.        ]
     [  20.           53.29969237   20.        ]
     [  30.           72.81704797   20.        ]
     [  40.           92.31053162   20.        ]
     [  50.          111.74134064   20.        ]
     [  60.          130.80209944   20.        ]
     [  70.           96.07373979   20.        ]
     [  80.           59.76966519   20.        ]
     [  90.           27.97189331   20.        ]
     [ 100.            0.           20.        ]]
    """

    ll_num = lut.shape[0]
    hh_num = lut.shape[1]
    ll_idx_float = lh_array[..., 0] / lightness_max * (ll_num - 1)
    ll_low_idx = np.int32(np.floor(ll_idx_float))
    ll_low_idx[ll_low_idx >= ll_num] = ll_num - 1
    ll_high_idx = ll_low_idx + 1
    ll_high_idx[ll_high_idx >= ll_num] = ll_num - 1
    # print(ll_low_idx)
    coef_after_shape = (ll_low_idx.shape[0], 1)
    ll_coef = (ll_idx_float - ll_low_idx).reshape(coef_after_shape)

    hh_idx_float = lh_array[..., 1] / 360 * (hh_num - 1)
    hh_low_idx = np.int32(np.floor(hh_idx_float))
    hh_high_idx = hh_low_idx + 1
    hh_high_idx[hh_high_idx >= hh_num] = hh_num - 1
    # print(hh_low_idx)
    coef_after_shape = (hh_low_idx.shape[0], 1)
    hh_coef = (hh_idx_float - hh_low_idx).reshape(coef_after_shape)

    # interpolate Lightness direction
    # print(lut[ll_low_idx, hh_low_idx].shape)
    # print(ll_idx_float - ll_low_idx)
    intp_l_hue_low = lut[ll_low_idx, hh_low_idx]\
        + (lut[ll_high_idx, hh_low_idx] - lut[ll_low_idx, hh_low_idx])\
        * ll_coef
    intp_l_hue_high = lut[ll_low_idx, hh_high_idx]\
        + (lut[ll_high_idx, hh_high_idx] - lut[ll_low_idx, hh_high_idx])\
        * ll_coef

    intp_lch = intp_l_hue_low\
        + (intp_l_hue_high - intp_l_hue_low) * hh_coef

    # print(intp_l_hue_low)
    # print(intp_l_hue_high)
    # print(intp_val)

    return intp_lch


def calc_cusp_specific_hue(lut, hue, lightness_max=100, ych=False):
    """
    calc gamut's cusp using lut.

    Parameters
    ----------
    lut : ndarray
        A gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    hue : float
        A Hue value. range is 0.0 - 360.0
    lightness_max : float
        maximum value of lightness

    Returns
    -------
    cusp : ndarray
        gamut's cusp.
        [Lightness, Chroma, Hue].

    Examples
    --------
    >>> calc_cusp_specific_hue(
    ...     lut=np.load("./lut/lut_sample_11_9_8192_ITU-R BT.2020.npy"),
    ...     hue=20)
    [  60.          130.80209944   20.        ]
    """
    l_num = lut.shape[0]
    if ych:
        ll_base = np.linspace(lightness_max, 0, l_num)
    else:
        ll_base = np.linspace(0, lightness_max, l_num)
    hh_base = np.ones_like(ll_base) * hue

    lh_array = tstack([ll_base, hh_base])

    lch = get_gamut_boundary_lch_from_lut(
        lut=lut, lh_array=lh_array, lightness_max=lightness_max)
    max_cc_idx = np.argmax(lch[..., 1])

    return lch[max_cc_idx]


def calc_cusp_specific_hue_for_YCH(lut, hue, lightness_max=100):
    """
    calc gamut's cusp using lut.

    Parameters
    ----------
    lut : ndarray
        A gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    hue : float
        A Hue value. range is 0.0 - 360.0
    lightness_max : float
        maximum value of lightness

    Returns
    -------
    cusp : ndarray
        gamut's cusp.
        [Lightness, Chroma, Hue].

    Examples
    --------
    >>> calc_cusp_specific_hue(
    ...     lut=np.load("./lut/lut_sample_11_9_8192_ITU-R BT.2020.npy"),
    ...     hue=20)
    [  60.          130.80209944   20.        ]
    """
    l_num = lut.shape[0]
    ll_base = np.linspace(lightness_max, 0, l_num)
    hh_base = np.ones_like(ll_base) * hue

    lh_array = tstack([ll_base, hh_base])

    lch = get_gamut_boundary_lch_from_lut(
        lut=lut, lh_array=lh_array, lightness_max=lightness_max)

    next = lch[2:-1, 1]
    before = lch[1:-2, 1]
    rate = (next - before) / before
    # for idx, val in enumerate(rate):
    #     print(idx, val)
    # print(np.argmin(rate))
    # print(next.shape, before.shape)
    max_cc_idx = np.argmin(rate) + 1
    # print(max_cc_idx)
    # max_cc_idx = np.argmax(lch[..., 1])
    # print(max_cc_idx)

    return lch[max_cc_idx]


def calc_l_focal_specific_hue(
        inner_lut, outer_lut, hue, maximum_l_focal=90, minimum_l_focal=50,
        lightness_max=100):
    """
    calc L_focal value

    Parameters
    ----------
    inner_lut : ndarray
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    outer_lut : ndarray
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    hue : float
        A Hue value. range is 0.0 - 360.0
    maximum_l_focal : float
        A maximum L_focal value.
        This is a parameter to prevent the data from changing
        from l_focal to cups transitioning to Out-of-Gamut.
    minimum_l_focal : float
        A maximum L_focal value.
        This is a parameter to prevent the data from changing
        from l_focal to cups transitioning to Out-of-Gamut.
    lightness_max : float
        maximum value of lightness

    Returns
    -------
    L_focal : ndarray
        L_focal value.
        [Lightness, Chroma, Hue].

    Examples
    --------
    >>> inner_lut = np.load("./lut/lut_sample_11_9_8192_ITU-R BT.709.npy")
    >>> outer_lut = np.load("./lut/lut_sample_11_9_8192_ITU-R BT.2020.npy")
    >>> calc_l_focal_specific_hue(inner_lut, outer_lut, 20)
    [ 32.7255767   0.         20.       ]
    """
    inner_cups = calc_cusp_specific_hue(
        lut=inner_lut, hue=hue, lightness_max=lightness_max)
    outer_cups = calc_cusp_specific_hue(
        lut=outer_lut, hue=hue, lightness_max=lightness_max)

    x1 = inner_cups[1]
    y1 = inner_cups[0]
    x2 = outer_cups[1]
    y2 = outer_cups[0]

    if x1 != x2:
        l_focal = (y2 - y1) / (x2 - x1) * (-x1) + y1
    else:
        l_focal = y1

    if l_focal > maximum_l_focal:
        l_focal = maximum_l_focal
    if l_focal < minimum_l_focal:
        l_focal = minimum_l_focal

    return np.array([l_focal, 0, hue])


def calc_chroma_boundary_specific_ligheness_jzazbz(
        lightness, hue_sample, cs_name, peak_luminance):
    """
    parameters
    ----------
    lightness : float
        lightness value(Jzazbz). range is 0.0 - 1.0.
    hue_sample : int
        Sample number of the Hue
    cs_name : string
        A color space name. ex. "ITU-R BT.709", "ITU-R BT.2020"

    Examples
    --------
    >>> boundary_jch = calc_chroma_boundary_specific_ligheness_jzazbz(
    ...     lightness=0.5, hue_sample=16, cs_name=cs.BT2020,
    ...     peak_luminance=10000)
    [[  5.00000000e-01   2.72627831e-01   0.00000000e+00]
     [  5.00000000e-01   2.96944618e-01   2.40000000e+01]
     [  5.00000000e-01   3.19167137e-01   4.80000000e+01]
     [  5.00000000e-01   2.51322746e-01   7.20000000e+01]
     [  5.00000000e-01   2.41002083e-01   9.60000000e+01]
     [  5.00000000e-01   2.76854515e-01   1.20000000e+02]
     [  5.00000000e-01   3.99024010e-01   1.44000000e+02]
     [  5.00000000e-01   2.64456749e-01   1.68000000e+02]
     [  5.00000000e-01   2.32390404e-01   1.92000000e+02]
     [  5.00000000e-01   2.51740456e-01   2.16000000e+02]
     [  5.00000000e-01   3.38995934e-01   2.40000000e+02]
     [  5.00000000e-01   3.09918404e-01   2.64000000e+02]
     [  5.00000000e-01   2.71250725e-01   2.88000000e+02]
     [  5.00000000e-01   2.59991646e-01   3.12000000e+02]
     [  5.00000000e-01   2.63157845e-01   3.36000000e+02]
     [  5.00000000e-01   2.72627831e-01   3.60000000e+02]]
    """
    # lch --> rgb

    r_val_init = 1.0
    trial_num = 30

    hue = np.linspace(0, 2*np.pi, hue_sample)
    r_val = r_val_init * np.ones_like(hue)
    jj = lightness * np.ones_like(hue)

    for t_idx in range(trial_num):
        aa = r_val * np.cos(hue)
        bb = r_val * np.sin(hue)
        jzazbz = tstack((jj, aa, bb))
        large_xyz = jzazbz_to_large_xyz(jzazbz)
        rgb_luminance = XYZ_to_RGB(
            large_xyz, cs.D65, cs.D65,
            RGB_COLOURSPACES[cs_name].matrix_XYZ_to_RGB)

        ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance/peak_luminance)
        ok_idx = np.logical_not(ng_idx)
        add_sub = r_val_init / (2 ** (t_idx + 1))
        r_val[ok_idx] = r_val[ok_idx] + add_sub
        r_val[~ok_idx] = r_val[~ok_idx] - add_sub

    jzczhz = tstack([jj, r_val, np.rad2deg(hue)])

    return jzczhz


def calc_chroma_boundary_specific_ligheness_jzazbz_type2(
        jj, chroma_sample, hue_num, cs_name, luminance=10000, **kwargs):
    """
    parameters
    ----------
    ll : float
        Jz value
    chroma_sample : int
        Sample number of the Chroma
        This value is related to accuracy.
    hue_num : int
        Sample number of the Hue
    cs_name : string
        A color space name. ex. "ITU-R BT.709", "ITU-R BT.2020"
    luminance : float
        A peak luminance

    Examples
    --------
    >>> boundary_jch = calc_chroma_boundary_specific_ligheness_jzazbz(
    ...     jj=0.5, chroma_sample=16384, hue_num=16,
    ...     cs_name=cs.BT2020, luminance=10000)
    [[  5.00000000e-01   2.72599646e-01   0.00000000e+00]
     [  5.00000000e-01   2.96923640e-01   2.40000000e+01]
     [  5.00000000e-01   3.19141793e-01   4.80000000e+01]
     [  5.00000000e-01   2.51297076e-01   7.20000000e+01]
     [  5.00000000e-01   2.40981505e-01   9.60000000e+01]
     [  5.00000000e-01   2.76841848e-01   1.20000000e+02]
     [  5.00000000e-01   3.99011170e-01   1.44000000e+02]
     [  5.00000000e-01   2.64450955e-01   1.68000000e+02]
     [  5.00000000e-01   2.32375023e-01   1.92000000e+02]
     [  5.00000000e-01   2.51724348e-01   2.16000000e+02]
     [  5.00000000e-01   3.38979430e-01   2.40000000e+02]
     [  5.00000000e-01   3.09894403e-01   2.64000000e+02]
     [  5.00000000e-01   2.71226271e-01   2.88000000e+02]
     [  5.00000000e-01   2.59964597e-01   3.12000000e+02]
     [  5.00000000e-01   2.63138619e-01   3.36000000e+02]
     [  5.00000000e-01   2.72599646e-01   3.60000000e+02]]
    """
    # lch --> rgb
    chroma_max = JZAZBZ_CHROMA_MAX
    hue_max = 360
    hue_base = np.linspace(0, hue_max, hue_num)
    jj_base = np.ones_like(hue_base) * jj
    chroma_base = np.linspace(0, chroma_max, chroma_sample)
    hh = hue_base.reshape((hue_num, 1))\
        * np.ones_like(chroma_base).reshape((1, chroma_sample))
    cc = chroma_base.reshape((1, chroma_sample))\
        * np.ones_like(hue_base).reshape((hue_num, 1))
    jj = np.ones_like(hh) * jj

    jzczhz = tstack((jj, cc, hh))

    # delete
    del hh
    del cc
    del jj

    jzazbz = jzczhz_to_jzazbz(jzczhz)

    large_xyz = jzazbz_to_large_xyz(jzazbz)
    del jzazbz

    rgb_luminance = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[cs_name].matrix_XYZ_to_RGB)
    del large_xyz

    ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance / luminance)
    del rgb_luminance

    chroma_array = np.zeros(hue_num)
    for h_idx in range(hue_num):
        chroma_ng_idx_array = np.where(ng_idx[h_idx] > 0)
        chroma_ng_idx = np.min(chroma_ng_idx_array)
        chroma_ng_idx = chroma_ng_idx - 1 if chroma_ng_idx > 0 else 0
        chroma_array[h_idx] = chroma_ng_idx / (chroma_sample - 1) * chroma_max

    jzczhz = tstack([jj_base, chroma_array, hue_base])

    return jzczhz


def calc_l_focal_specific_hue_jzazbz(
        inner_lut, outer_lut, hue, maximum_l_focal=90, minimum_l_focal=50):
    """
    calc L_focal value

    Parameters
    ----------
    inner_lut : TyLchLut
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    outer_lut : TyLchLut
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    hue : float
        A Hue value. range is 0.0 - 360.0
    maximum_l_focal : float
        A maximum L_focal value.
        This is a parameter to prevent the data from changing
        from l_focal to cups transitioning to Out-of-Gamut.
    minimum_l_focal : float
        A maximum L_focal value.
        This is a parameter to prevent the data from changing
        from l_focal to cups transitioning to Out-of-Gamut.

    Returns
    -------
    L_focal : ndarray
        L_focal value.
        [Lightness, Chroma, Hue].

    Examples
    --------
    """

    inner_cups = calc_cusp_specific_hue(
        lut=inner_lut.lut, hue=hue, lightness_max=inner_lut.ll_max)
    outer_cups = calc_cusp_specific_hue(
        lut=outer_lut.lut, hue=hue, lightness_max=outer_lut.ll_max)

    x1 = inner_cups[1]
    y1 = inner_cups[0]
    x2 = outer_cups[1]
    y2 = outer_cups[0]

    if x1 != x2:
        l_focal = (y2 - y1) / (x2 - x1) * (-x1) + y1
    else:
        l_focal = y1

    if l_focal > maximum_l_focal:
        l_focal = maximum_l_focal
    if l_focal < minimum_l_focal:
        l_focal = minimum_l_focal

    return np.array([l_focal, 0, hue])


def create_focal_point_lut_jzazbz(
        inner_lut, outer_lut, maximum_l_focal=0.8, minimum_l_focal=0.35):
    """
    Parameters
    ----------
    inner_lut : TyLchLut
        inner lut
    outer_lut : TyLchLut
        outer lut
    maximum_l_focal : float
        maximum focal value
    minimum_l_focal : float
        minimum focal value
    Examples
    --------
    >>> inner_lut = np.load(
    ...     "./lut/JzChz_gb-lut_ITU-R BT.709_10000nits_jj-256_hh-16.npy")
    >>> outer_lut = np.load(
    ...     "./lut/JzChz_gb-lut_ITU-R BT.2020_10000nits_jj-256_hh-16.npy")
    >>> focal_array = create_focal_point_lut_jzazbz(inner_lut, outer_lut)
    [[  3.75796277e-01   0.00000000e+00   0.00000000e+00]
     [  3.67347037e-01   0.00000000e+00   2.40000000e+01]
     [  5.54695842e-02   0.00000000e+00   4.80000000e+01]
     [  4.05505556e-01   0.00000000e+00   7.20000000e+01]
     [  7.44164104e-01   0.00000000e+00   9.60000000e+01]
     [  8.43137255e-01   0.00000000e+00   1.20000000e+02]
     [  8.68700250e-01   0.00000000e+00   1.44000000e+02]
     [  8.87831459e-01   0.00000000e+00   1.68000000e+02]
     [  9.00159536e-01   0.00000000e+00   1.92000000e+02]
     [  9.21095637e-01   0.00000000e+00   2.16000000e+02]
     [  8.36714636e-01   0.00000000e+00   2.40000000e+02]
     [  4.13055665e-01   0.00000000e+00   2.64000000e+02]
     [  6.05061364e-01   0.00000000e+00   2.88000000e+02]
     [  7.06398497e-01   0.00000000e+00   3.12000000e+02]
     [  3.39021253e-01   0.00000000e+00   3.36000000e+02]
     [  3.75796277e-01   0.00000000e+00   3.60000000e+02]]
    """
    hue_array = inner_lut.lut[0, :, 2]
    focal_array = []
    for hue in hue_array:
        focal = calc_l_focal_specific_hue_jzazbz(
            inner_lut=inner_lut, outer_lut=outer_lut, hue=hue,
            maximum_l_focal=maximum_l_focal,
            minimum_l_focal=minimum_l_focal)
        focal_array.append(focal)
    focal_array = np.array(focal_array)

    return focal_array


class TyLchLut():
    """
    Toru Yoshihara's Lightness-Chroma-Hue Lut.
    """
    def __init__(self, lut):
        self.lut = lut
        self.ll_min = lut[0, 0, 0]
        self.ll_max = lut[-1, 0, 0]
        print(f"LUT loaded. min, max = {self.ll_min}, {self.ll_max}")

    def interpolate(self, lh_array):
        return get_gamut_boundary_lch_from_lut(
            lut=self.lut, lh_array=lh_array, lightness_max=self.ll_max)

    def get_cusp(self, hue, ych=False):
        """
        calc gamut's cusp using lut.

        Parameters
        ----------
        hue : float
            A Hue value. range is 0.0 - 360.0
        lightness_max : float
            maximum value of lightness

        Returns
        -------
        cusp : ndarray
            gamut's cusp.
            [Lightness, Chroma, Hue].

        Examples
        --------
        >>> calc_cusp_specific_hue(
        ...     lut=np.load("./lut/lut_sample_11_9_8192_ITU-R BT.2020.npy"),
        ...     hue=20)
        [  60.          130.80209944   20.        ]
        """
        if ych:
            cusp_lch = calc_cusp_specific_hue_for_YCH(
                lut=self.lut, hue=hue, lightness_max=self.ll_max)
        else:
            cusp_lch = calc_cusp_specific_hue(
                lut=self.lut, hue=hue, lightness_max=self.ll_max)

        return cusp_lch

    def get_cusp_without_intp(self, hue):
        """
        calc gamut's cusp without interpolation from lut.

        Parameters
        ----------
        hue : float
            A Hue value. range is 0.0 - 360.0
        lightness_max : float
            maximum value of lightness

        Returns
        -------
        cusp : ndarray
            gamut's cusp.
            [Lightness, Chroma, Hue].

        Examples
        --------
        >>> calc_cusp_specific_hue(
        ...     lut=np.load("./lut/lut_sample_11_9_8192_ITU-R BT.2020.npy"),
        ...     hue=20)
        [  60.          130.80209944   20.        ]
        """
        hue_sample = self.lut.shape[1]
        hue_idx = int(round(hue / 360 * hue_sample))
        # print(f"hue_sample={hue_sample}, hue_idx={hue_idx}")

        lch = self.lut[:, hue_idx, :]
        max_cc_idx = np.argmax(lch[..., 1])
        cusp_lch = lch[max_cc_idx]
        # print(f"cusp_lch={cusp_lch}")

        return cusp_lch


def make_cielab_gb_lut_fname_method_b(
        color_space_name, lightness_num, hue_num):
    fname = f"./lut/cielab_gb-lut_method_b_{color_space_name}_"
    fname += f"ll-{lightness_num}_hh-{hue_num}.npy"

    return fname


def make_cielab_gb_lut_fname_method_c(
        color_space_name, lightness_num, hue_num):
    fname = f"./lut/cielab_gb-lut_method_c_{color_space_name}_"
    fname += f"ll-{lightness_num}_hh-{hue_num}.npy"

    return fname


def make_jzazbz_gb_lut_fname(
        color_space_name, luminance, lightness_num, hue_num):
    fname = f"./lut/JzChz_gb-lut_type3_{color_space_name}_"
    fname += f"{luminance}nits_jj-{lightness_num}_"
    fname += f"hh-{hue_num}.npy"

    return fname


def make_jzazbz_gb_lut_fname_methodb_b(
        color_space_name, luminance, lightness_num, hue_num):
    fname = f"./lut/JzChz_gb-lut_method_b_{color_space_name}_"
    fname += f"{luminance}nits_jj-{lightness_num}_"
    fname += f"hh-{hue_num}.npy"

    return fname


def make_jzazbz_gb_lut_fname_method_c(
        color_space_name, luminance,
        lightness_num=L_SAMPLE_DEFAULT, hue_num=H_SAMPLE_DEFAULT):
    fname = f"./lut/JzChz_gb-lut_method_c_{color_space_name}_"
    fname += f"{luminance}nits_jj-{lightness_num}_"
    fname += f"hh-{hue_num}.npy"

    return fname


def make_jzazbz_gb_lut_fname_old(
        color_space_name, luminance, lightness_num, hue_num):
    fname = f"./lut/JzChz_gb-lut_type1_{color_space_name}_"
    fname += f"{luminance}nits_jj-{lightness_num}_"
    fname += f"hh-{hue_num}.npy"

    return fname


def make_jzazbz_focal_lut_fname(
        luminance, lightness_num, hue_num, prefix="BT709_BT2020"):
    fname = f"./lut/JzChz_focal-lut_{prefix}_"
    fname += f"{luminance}nits_jj-{lightness_num}_"
    fname += f"hh-{hue_num}.npy"

    return fname


def make_jzazbz_focal_lut_fname_wo_lpf(
        luminance, lightness_num, hue_num, prefix="BT709_BT2020"):
    fname = f"./lut/JzChz_focal-lut_wo_lpf_{prefix}_"
    fname += f"{luminance}nits_jj-{lightness_num}_"
    fname += f"hh-{hue_num}.npy"

    return fname


def create_jzazbz_gamut_boundary_lut(
        hue_sample=256, lightness_sample=256,
        color_space_name=cs.BT2020, luminance=10000):
    """
    Parameters
    ----------
    hue_sample : int
        The number of hue
    lightness_sample : int
        The number of lightness
    color_space_name : strings
        color space name for colour.RGB_COLOURSPACES
    luminance : float
        peak luminance for Jzazbz color space
    """

    lut = []
    met = MeasureExecTime()
    met.start()
    for j_val in np.linspace(0, 1, lightness_sample):
        print(f"j_val = {j_val:.3f}, ", end="")
        met.lap()
        jzczhz = calc_chroma_boundary_specific_ligheness_jzazbz(
            lightness=j_val, hue_sample=hue_sample,
            cs_name=color_space_name, peak_luminance=luminance)
        lut.append(jzczhz)
    met.end()
    lut = np.array(lut)

    fname = make_jzazbz_gb_lut_fname_old(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    np.save(fname, np.float32(lut))


def thread_wrapper_create_jzazbz_gamut_boundary_lut_type2(args):
    jzczhz = calc_chroma_boundary_specific_ligheness_jzazbz_type2(**args)

    hue_num = args['hue_num']
    hue_plane_size = hue_num * 3
    l_idx = args['l_idx']

    base_addr = l_idx * hue_plane_size

    for h_idx in range(hue_num):
        addr = base_addr + h_idx * 3
        shared_array[addr:addr+3] = np.float32(jzczhz[h_idx])


def create_jzazbz_gamut_boundary_lut_type2(
        hue_sample=256, lightness_sample=256, chroma_sample=32768,
        color_space_name=cs.BT2020, luminance=10000):
    """
    Parameters
    ----------
    hue_sample : int
        The number of hue
    lightness_sample : int
        The number of lightness
    chroma_sample : int
        The number of chroma.
        This value is related to accuracy.
    color_space_name : strings
        color space name for colour.RGB_COLOURSPACES
    luminance : float
        peak luminance for Jzazbz color space
    """

    total_process_num = lightness_sample
    # block_process_num = cpu_count()
    block_process_num = 16  # for 32768 sample
    block_num = int(round(total_process_num / block_process_num + 0.5))
    max_jz = large_xyz_to_jzazbz(xy_to_XYZ(cs.D65) * luminance)[0]
    print(f"max_Jz = {max_jz}")

    mtime = MeasureExecTime()
    mtime.start()
    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            d = dict(
                jj=l_idx/(lightness_sample-1) * max_jz,
                chroma_sample=chroma_sample,
                hue_num=hue_sample, cs_name=color_space_name,
                luminance=luminance, l_idx=l_idx)
            args.append(d)
            # thread_wrapper_create_jzazbz_gamut_boundary_lut_type2(d)
        with Pool(block_process_num) as pool:
            pool.map(
                thread_wrapper_create_jzazbz_gamut_boundary_lut_type2, args)
            mtime.lap()
        mtime.lap()
    mtime.end()

    lut = np.array(
        shared_array[:lightness_sample*hue_sample*3]).reshape(
            (lightness_sample, hue_sample, 3))

    fname = make_jzazbz_gb_lut_fname_methodb_b(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    np.save(fname, np.float32(lut))


def thread_wrapper_create_jzazbz_gamut_boundary_lut_type3(args):
    calc_chroma_boundary_specific_ligheness_jzazbz_type3(**args)

    # h_min_idx = args['hue_min_idx']
    # h_max_idx = args['hue_max_idx']

    # for h_idx in range(h_min_idx, h_max_idx):
    #     addr = h_idx * 3
    #     shared_array[addr:addr+3] = np.float32(jzczhz[h_idx-h_min_idx])


def calc_chroma_boundary_specific_ligheness_jzazbz_type3(
        jj, chroma_sample, hue_min_idx, hue_max_idx, hue_num, cs_name,
        luminance=10000, **kwargs):
    """
    parameters
    ----------
    ll : float
        Jz value
    chroma_sample : int
        Sample number of the Chroma
        This value is related to accuracy.
    hue_num : int
        Sample number of the Hue
    cs_name : string
        A color space name. ex. "ITU-R BT.709", "ITU-R BT.2020"
    luminance : float
        A peak luminance

    Examples
    --------
    >>> boundary_jch = calc_chroma_boundary_specific_ligheness_jzazbz(
    ...     jj=0.5, chroma_sample=16384, hue_num=16,
    ...     cs_name=cs.BT2020, luminance=10000)
    [[  5.00000000e-01   2.72599646e-01   0.00000000e+00]
     [  5.00000000e-01   2.96923640e-01   2.40000000e+01]
     [  5.00000000e-01   3.19141793e-01   4.80000000e+01]
     [  5.00000000e-01   2.51297076e-01   7.20000000e+01]
     [  5.00000000e-01   2.40981505e-01   9.60000000e+01]
     [  5.00000000e-01   2.76841848e-01   1.20000000e+02]
     [  5.00000000e-01   3.99011170e-01   1.44000000e+02]
     [  5.00000000e-01   2.64450955e-01   1.68000000e+02]
     [  5.00000000e-01   2.32375023e-01   1.92000000e+02]
     [  5.00000000e-01   2.51724348e-01   2.16000000e+02]
     [  5.00000000e-01   3.38979430e-01   2.40000000e+02]
     [  5.00000000e-01   3.09894403e-01   2.64000000e+02]
     [  5.00000000e-01   2.71226271e-01   2.88000000e+02]
     [  5.00000000e-01   2.59964597e-01   3.12000000e+02]
     [  5.00000000e-01   2.63138619e-01   3.36000000e+02]
     [  5.00000000e-01   2.72599646e-01   3.60000000e+02]]
    """
    # lch --> rgb
    chroma_max = JZAZBZ_CHROMA_MAX
    hue_max = 360
    hue_base_all = np.linspace(0, hue_max, hue_num)
    hue_base = hue_base_all[hue_min_idx:hue_max_idx]

    hue_base_limited = len(hue_base)

    jj_base = np.ones_like(hue_base) * jj
    chroma_base = np.linspace(0, chroma_max, chroma_sample)
    hh = hue_base.reshape((hue_base_limited, 1))\
        * np.ones_like(chroma_base).reshape((1, chroma_sample))
    cc = chroma_base.reshape((1, chroma_sample))\
        * np.ones_like(hue_base).reshape((hue_base_limited, 1))
    jj = np.ones_like(hh) * jj

    jzczhz = tstack((jj, cc, hh))

    # delete
    del hh
    del cc
    del jj

    jzazbz = jzczhz_to_jzazbz(jzczhz)

    large_xyz = jzazbz_to_large_xyz(jzazbz)
    del jzazbz

    rgb_luminance = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[cs_name].matrix_XYZ_to_RGB)
    del large_xyz

    ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance / luminance)
    del rgb_luminance

    chroma_array = np.zeros(hue_base_limited)
    for h_idx in range(hue_base_limited):
        chroma_ng_idx_array = np.where(ng_idx[h_idx] > 0)
        chroma_ng_idx = np.min(chroma_ng_idx_array)
        chroma_ng_idx = chroma_ng_idx - 1 if chroma_ng_idx > 0 else 0
        chroma_array[h_idx] = chroma_ng_idx / (chroma_sample - 1) * chroma_max

    jzczhz = tstack([jj_base, chroma_array, hue_base])

    h_min_idx = hue_min_idx
    h_max_idx = hue_max_idx

    for h_idx in range(h_min_idx, h_max_idx):
        addr = h_idx * 3
        shm_buf[addr:addr+3] = np.float32(jzczhz[h_idx-h_min_idx])

    return jzczhz


def create_jzazbz_gamut_boundary_lut_type3(
        hue_sample=256, lightness_sample=256, chroma_sample=32768,
        color_space_name=cs.BT2020, luminance=10000):
    """
    Parameters
    ----------
    hue_sample : int
        The number of hue
    lightness_sample : int
        The number of lightness
    chroma_sample : int
        The number of chroma.
        This value is related to accuracy.
    color_space_name : strings
        color space name for colour.RGB_COLOURSPACES
    luminance : float
        peak luminance for Jzazbz color space
    """
    hue_div_num = 4
    hue_idx_step = 16

    total_process_num = hue_sample // hue_div_num
    if (total_process_num != int(hue_sample / hue_div_num)):
        print("invalid hue_sample!!")
        sys.exit(1)
    block_process_num = 16
    block_num = int(
        round(total_process_num / (block_process_num * hue_idx_step) + 0.5))
    max_jz = large_xyz_to_jzazbz(xy_to_XYZ(cs.D65) * luminance)[0]
    print(f"max_Jz = {max_jz}")
    lut = np.zeros((lightness_sample, hue_sample, 3))

    mtime = MeasureExecTime()
    mtime.start()
    for l_idx in range(lightness_sample):
        jj = l_idx/(lightness_sample-1) * max_jz
        for h_block_idx in range(hue_div_num):
            h_idx_base = h_block_idx * total_process_num
            for b_idx in range(block_num):
                args = []
                for p_idx in range(block_process_num):
                    local_h_idx = b_idx * block_process_num * hue_idx_step\
                        + p_idx * hue_idx_step
                    h_idx_st = h_idx_base + local_h_idx
                    h_idx_ed = h_idx_st + hue_idx_step
                    msg = f"l_idx={l_idx}, h_b_idx={h_block_idx}, "
                    msg += f"b_idx={b_idx}, p_idx={p_idx}, "
                    msg += f"h_idx_st={h_idx_st}, h_idx_ed={h_idx_ed}"
                    print(msg)
                    if h_idx_ed > h_idx_base + total_process_num:
                        break
                    d = dict(
                        jj=jj, hue_min_idx=h_idx_st, hue_max_idx=h_idx_ed,
                        chroma_sample=chroma_sample,
                        hue_num=hue_sample, cs_name=color_space_name,
                        luminance=luminance, l_idx=l_idx)
                    args.append(d)
                    # thread_wrapper_create_jzazbz_gamut_boundary_lut_type3(d)
                with Pool(block_process_num) as pool:
                    pool.map(
                        thread_wrapper_create_jzazbz_gamut_boundary_lut_type3,
                        args)
                print("p roop finished")
                print(f"block_num={block_num}, hue_div_num={hue_div_num}")
                mtime.lap()
        lut[l_idx] = np.array(
            shm_buf[:hue_sample*3]).reshape(1, hue_sample, 3)

    fname = make_jzazbz_gb_lut_fname(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    np.save(fname, np.float32(lut))


def low_pass_filter2(x, nn=4, wn=0.25):
    b1, a1 = signal.bessel(nn, wn, "low")
    result = signal.filtfilt(b1, a1, x)

    return result


def apply_lpf_to_focal_lut(
        luminance, lightness_num, hue_num, prefix="BT709-BT2020",
        maximum_l_focal=1.0, minimum_l_focal=0.0, wn=0.06,
        inner_cs_name=cs.BT709, outer_cs_name=cs.BT2020):
    if inner_cs_name == cs.BT709:
        inner_lut_name = make_jzazbz_gb_lut_fname(
            color_space_name=inner_cs_name, luminance=luminance,
            lightness_num=lightness_num, hue_num=hue_num)
    else:
        inner_lut_name = make_jzazbz_gb_lut_fname_old(
            color_space_name=inner_cs_name, luminance=luminance,
            lightness_num=lightness_num, hue_num=hue_num)

    outer_lut_name = make_jzazbz_gb_lut_fname_old(
        color_space_name=outer_cs_name, luminance=luminance,
        lightness_num=lightness_num, hue_num=hue_num)
    inner_lut = TyLchLut(lut=np.load(inner_lut_name))
    outer_lut = TyLchLut(lut=np.load(outer_lut_name))

    focal_array_wo_lpf = create_focal_point_lut_jzazbz(
        inner_lut=inner_lut, outer_lut=outer_lut,
        maximum_l_focal=maximum_l_focal, minimum_l_focal=minimum_l_focal)

    ll = focal_array_wo_lpf[..., 0]
    ll_new = low_pass_filter2(
        ll, nn=4, wn=wn)

    focal_array_w_lpf = tstack(
        [ll_new, focal_array_wo_lpf[..., 1], focal_array_wo_lpf[..., 2]])

    focal_lut_w_lpf_name = make_jzazbz_focal_lut_fname(
        luminance=luminance,
        lightness_num=lightness_num, hue_num=hue_num,
        prefix=prefix)
    focal_lut_wo_lpf_name = make_jzazbz_focal_lut_fname_wo_lpf(
        luminance=luminance,
        lightness_num=lightness_num, hue_num=hue_num,
        prefix=prefix)
    np.save(focal_lut_w_lpf_name, focal_array_w_lpf)
    np.save(focal_lut_wo_lpf_name, focal_array_wo_lpf)


def get_focal_point_from_lut(focal_point_lut, h_val):
    """
    Parameters
    ----------
    focal_point_lut : ndarray
        lut. shape is (N, 3).
    h_val : ndarray
        hue value. unit is degree.
        ex. h_val = 120, h_val = 359.
    """
    xx = focal_point_lut[..., 2]  # hue
    yy = focal_point_lut[..., 0]  # lightness
    func = interpolate.interp1d(xx, yy)
    focal_point = func(h_val)

    return focal_point


def make_Ych_gb_lut_fname(
        color_space_name, lightness_num, hue_num):
    fname = f"./lut/Ych_gb-lut_{color_space_name}_"
    fname += f"Y-{lightness_num}_"
    fname += f"hh-{hue_num}.npy"

    return fname


def create_Ych_gamut_boundary_lut(
        hue_sample=256, lightness_sample=256, chroma_sample=32768,
        color_space_name=cs.BT2020):
    """
    Parameters
    ----------
    hue_sample : int
        The number of hue
    lightness_sample : int
        The number of lightness
    chroma_sample : int
        The number of chroma.
        This value is related to accuracy.
    color_space_name : strings
        color space name for colour.RGB_COLOURSPACES
    """

    total_process_num = lightness_sample
    # block_process_num = cpu_count()
    block_process_num = 16  # for 32768 sample
    # block_process_num = 24  # for 16384 sample
    block_num = int(round(total_process_num / block_process_num + 0.5))
    max_Y = 1.0
    print(f"max_Y = {max_Y}")

    mtime = MeasureExecTime()
    mtime.start()
    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            d = dict(
                large_y=l_idx/(lightness_sample-1) * max_Y,
                chroma_sample=chroma_sample, hue_num=hue_sample,
                cs_name=color_space_name, l_idx=l_idx)
            args.append(d)
            # thread_wrapper_calc_chroma_boundary_specific_lightness_xyY(d)
        with Pool(block_process_num) as pool:
            pool.map(
                thread_wrapper_calc_chroma_boundary_specific_lightness_xyY,
                args)
            mtime.lap()
        mtime.lap()
    mtime.end()

    lut = np.array(
        shared_array[:lightness_sample*hue_sample*3]).reshape(
            (lightness_sample, hue_sample, 3))

    fname = make_Ych_gb_lut_fname(
        color_space_name=color_space_name,
        lightness_num=lightness_sample, hue_num=hue_sample)
    np.save(fname, np.float32(lut))


def thread_wrapper_calc_chroma_boundary_specific_lightness_xyY(args):
    Ych = calc_chroma_boundary_specific_lightness_xyY(**args)

    hue_num = args['hue_num']
    hue_plane_size = hue_num * 3
    l_idx = args['l_idx']

    base_addr = l_idx * hue_plane_size

    for h_idx in range(hue_num):
        addr = base_addr + h_idx * 3
        shared_array[addr:addr+3] = np.float32(Ych[h_idx])


def calc_chroma_boundary_specific_lightness_xyY(
        large_y, chroma_sample, hue_num, cs_name, **kwargs):
    """
    parameters
    ----------
    large_y : float
        Y value
    chroma_sample : int
        Sample number of the Chroma
        This value is related to accuracy.
    hue_num : int
        Sample number of the Hue
    cs_name : string
        A color space name. ex. "ITU-R BT.709", "ITU-R BT.2020"

    Examples
    --------
    >>> Ych = calc_chroma_boundary_specific_lightness_xyY(
    >>>     large_y=0.5, chroma_sample=16384, hue_num=16, cs_name=cs.BT2020)
    [[  5.00000000e-01   1.67002381e-01   0.00000000e+00]
     [  5.00000000e-01   2.64176280e-01   2.40000000e+01]
     [  5.00000000e-01   2.43606177e-01   4.80000000e+01]
     [  5.00000000e-01   2.69120430e-01   7.20000000e+01]
     [  5.00000000e-01   3.72642373e-01   9.60000000e+01]
     [  5.00000000e-01   3.06415186e-01   1.20000000e+02]
     [  5.00000000e-01   1.98864677e-01   1.44000000e+02]
     [  5.00000000e-01   1.68833547e-01   1.68000000e+02]
     [  5.00000000e-01   1.72617958e-01   1.92000000e+02]
     [  5.00000000e-01   1.31416712e-01   2.16000000e+02]
     [  5.00000000e-01   1.07916743e-01   2.40000000e+02]
     [  5.00000000e-01   1.07306354e-01   2.64000000e+02]
     [  5.00000000e-01   1.26106330e-01   2.88000000e+02]
     [  5.00000000e-01   1.14203748e-01   3.12000000e+02]
     [  5.00000000e-01   1.23908930e-01   3.36000000e+02]
     [  5.00000000e-01   1.67002381e-01   3.60000000e+02]]
    """
    chroma_max = 1.0
    hue_max = 360
    hue_base = np.linspace(0, hue_max, hue_num)
    chroma_base = np.linspace(0, chroma_max, chroma_sample)
    hh = hue_base.reshape((hue_num, 1))\
        * np.ones_like(chroma_base).reshape((1, chroma_sample))
    cc = chroma_base.reshape((1, chroma_sample))\
        * np.ones_like(hue_base).reshape((hue_num, 1))
    YY = np.ones_like(hh) * large_y
    YY_base = np.ones_like(hue_base) * large_y

    Ych = tstack((YY, cc, hh))

    # jzazbz = jzczhz_to_jzazbz(jzczhz)
    xyY = cs.Ych_to_xyY(Ych)

    # large_xyz = jzazbz_to_large_xyz(jzazbz)
    large_xyz = xyY_to_XYZ(xyY)

    rgb_luminance = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[cs_name].matrix_XYZ_to_RGB)

    ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance)

    chroma_array = np.zeros(hue_num)
    for h_idx in range(hue_num):
        if large_y > 0.0:
            chroma_ng_idx_array = np.where(ng_idx[h_idx] > 0)
            chroma_ng_idx = np.min(chroma_ng_idx_array)
            chroma_ng_idx = chroma_ng_idx - 1 if chroma_ng_idx > 0 else 0
            chroma_array[h_idx]\
                = chroma_ng_idx / (chroma_sample - 1) * chroma_max
        else:
            chroma_array[h_idx] = 0

    Ych = tstack([YY_base, chroma_array, hue_base])

    return Ych


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    """ Create Gamut Boundary LUTs"""
    # hue_sample = 1024
    # chroma_sample = 32768
    # ll_num = 1024
    # cs_name = cs.P3_D65
    # lut = calc_chroma_boundary_lut(
    #     lightness_sample=ll_num, chroma_sample=chroma_sample,
    #     hue_sample=hue_sample, cs_name=cs_name)
    # np.save(
    #     f"./lut/lut_sample_{ll_num}_{hue_sample}_{chroma_sample}_{cs_name}.npy",
    #     lut)


    """ check interpolation """
    # lut = np.load("./lut/lut_sample_25_100_8192.npy")
    # # l = 13
    # # h = 15
    # print(np.linspace(0, 100, 25))
    # print(np.linspace(0, 360, 100))
    # hh = np.array([14, 19])
    # ll = np.ones_like(hh) * 13
    # lh_array = tstack([ll, hh])
    # print(lh_array)
    # get_gamut_boundary_lch_from_lut(lut=lut, lh_array=lh_array)

    """ check cups """
    # calc_cusp_specific_hue(
    #     lut=np.load("./lut/lut_sample_11_9_8192_ITU-R BT.2020.npy"),
    #     hue=20)

    # inner_lut = np.load("./lut/lut_sample_11_9_8192_ITU-R BT.709.npy")
    # outer_lut = np.load("./lut/lut_sample_11_9_8192_ITU-R BT.2020.npy")
    # l_focal = calc_l_focal_specific_hue(inner_lut, outer_lut, 20)
    # print(l_focal)

    """ focal lut """
    # inner_lut = np.load(
    #     "./lut/JzChz_gb-lut_ITU-R BT.709_10000nits_jj-256_hh-16.npy")
    # outer_lut = np.load(
    #     "./lut/JzChz_gb-lut_ITU-R BT.2020_10000nits_jj-256_hh-16.npy")
    # focal_array = create_focal_point_lut_jzazbz(inner_lut, outer_lut)
    # print(focal_array)

    # boundary_jch = calc_chroma_boundary_specific_ligheness_jzazbz(
    #     lightness=0.5, hue_sample=16, cs_name=cs.BT2020,
    #     peak_luminance=10000)
    # print(boundary_jch)

    # lut = np.zeros((4, 9, 3))
    # hue = np.linspace(0, 360, 9)
    # # chroma_0 = np.ones_like(hue) * 0
    # # chroma_1 = np.ones_like(hue) * 4.0
    # # chroma_2 = np.ones_like(hue) * 8.0
    # # chroma_3 = np.ones_like(hue) * 16.0
    # chroma_0 = np.linspace(0, 1, len(hue)) * 0
    # chroma_1 = np.linspace(0, 1, len(hue)) * 4
    # chroma_2 = np.linspace(0, 1, len(hue)) * 8
    # chroma_3 = np.linspace(0, 1, len(hue)) * 16
    # lightness_0 = np.ones_like(hue) * 0
    # lightness_1 = np.ones_like(hue) * 20
    # lightness_2 = np.ones_like(hue) * 40
    # lightness_3 = np.ones_like(hue) * 60
    # lch_0 = tstack([lightness_0, chroma_0, hue])
    # lch_1 = tstack([lightness_1, chroma_1, hue])
    # lch_2 = tstack([lightness_2, chroma_2, hue])
    # lch_3 = tstack([lightness_3, chroma_3, hue])
    # lut[0] = lch_0
    # lut[1] = lch_1
    # lut[2] = lch_2
    # lut[3] = lch_3
    # print(lut)
    # lh_array = np.array([[55, 45 * 3/4]])
    # lch = get_gamut_boundary_lch_from_lut(
    #     lut=lut, lh_array=lh_array, lightness_max=60)
    # print(lch)

    # create_Ych_gamut_boundary_lut(
    #     hue_sample=16, lightness_sample=101, chroma_sample=1024,
    #     color_space_name=cs.BT2020)
