# -*- coding: utf-8 -*-
"""
create gamut boundary lut.
"""

# import standard libraries
import os
import ctypes

# import third-party libraries
import numpy as np
from colour.utilities import tstack
from colour import LCHab_to_Lab, Lab_to_XYZ, XYZ_to_RGB
from colour import RGB_COLOURSPACES
from multiprocessing import Pool, cpu_count, Array

# import my libraries
import color_space as cs
from common import MeasureExecTime
from jzazbz import jzazbz_to_large_xyz, jzczhz_to_jzazbz

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

L_SAMPLE_NUM_MAX = 1024
H_SAMPLE_NUM_MAX = 1024
COLOR_NUM = 3

shared_array_type = ctypes.c_float
shared_array_elem_size = ctypes.sizeof(shared_array_type)
shared_array = Array(
    typecode_or_type=shared_array_type,
    size_or_initializer=L_SAMPLE_NUM_MAX*H_SAMPLE_NUM_MAX*COLOR_NUM)

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


def is_out_of_gamut_rgb(rgb):
    min_val = -DELTA
    max_val = 1 + DELTA
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


def calc_chroma_boundary_specific_hue(
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
    chroma_max = 220
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


def thread_wrapper_calc_chroma_boundary_specific_hue(args):
    lightness_array, chroma_array = calc_chroma_boundary_specific_hue(**args)
    hue_array = np.ones_like(lightness_array) * args['hue']
    plane_lut = tstack([lightness_array, chroma_array, hue_array])
    ll_len = args['lightness_sample']
    hh_len = args['hue_sample']
    h_idx = args['hue_idx']

    hue_plane_size = hh_len * 3

    for l_idx in range(ll_len):
        addr = (hue_plane_size * l_idx) + (h_idx * 3)
        shared_array[addr:addr+3] = np.float32(plane_lut[l_idx])


def calc_chroma_boundary_lut(
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
    # block_process_num = cpu_count()
    block_process_num = 3  # for 32768 sample
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
            pool.map(thread_wrapper_calc_chroma_boundary_specific_hue, args)
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


def calc_cusp_specific_hue(lut, hue, lightness_max=100):
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
    ll_base = np.linspace(0, lightness_max, l_num)
    hh_base = np.ones_like(ll_base) * hue

    lh_array = tstack([ll_base, hh_base])

    lch = get_gamut_boundary_lch_from_lut(
        lut=lut, lh_array=lh_array, lightness_max=lightness_max)
    max_cc_idx = np.argmax(lch[..., 1])

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
    trial_num = 20

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
    chroma_max = 0.5
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
    jzazbz = jzczhz_to_jzazbz(jzczhz)

    large_xyz = jzazbz_to_large_xyz(jzazbz)
    rgb_luminance = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[cs_name].matrix_XYZ_to_RGB)
    ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance / luminance)

    chroma_array = np.zeros(hue_num)
    for h_idx in range(hue_num):
        chroma_ng_idx_array = np.where(ng_idx[h_idx] > 0)
        chroma_ng_idx = np.min(chroma_ng_idx_array)
        chroma_ng_idx = chroma_ng_idx - 1 if chroma_ng_idx > 0 else 0
        chroma_array[h_idx] = chroma_ng_idx / (chroma_sample - 1) * chroma_max

    jzczhz = tstack([jj_base, chroma_array, hue_base])

    return jzczhz


def create_focal_point_lut_jzazbz(inner_lut, outer_lut):
    """
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
    hue_array = inner_lut[0, :, 2]
    focal_array = []
    for hue in hue_array:
        focal = calc_l_focal_specific_hue(
            inner_lut=inner_lut, outer_lut=outer_lut, hue=hue,
            maximum_l_focal=1.0, minimum_l_focal=0.0, lightness_max=1.0)
        focal_array.append(focal)
    focal_array = np.array(focal_array)

    return focal_array


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
