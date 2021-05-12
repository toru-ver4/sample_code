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
import light

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


# def calc_chroma_boundary_lut(
#         lightness_sample, chroma_sample, chroma_max, hue_num, cs_name):
#     """
#     parameters
#     ----------
#     lightness_sample : int
#         Sample number of the Lightness
#         Lightness range is 0.0 - 100.0
#     chroma_sample : int
#         Sample number of the Chroma
#     chroma_max : float
#         The maximum value of the Chroma search range.
#     hue_num : int
#         Sample number of the Hue
#     cs_name : string
#         A color space name. ex. "ITU-R BT.709", "ITU-R BT.2020"
#     """
#     # create buffer (2D-LUT)
#     lut = np.zeros((lightness_sample, hue_num, 3))
#     mtime = MeasureExecTime()
#     mtime.start()
#     for l_idx in range(lightness_sample):
#         ll = l_idx / (lightness_sample - 1) * 100
#         print(f"l_idx={l_idx}, l_val={ll}")
#         hue_array, chroma_array = calc_chroma_boundary_specific_l(
#             ll=ll, chroma_sample=chroma_sample, chroma_max=chroma_max,
#             hue_num=hue_num, cs_name=cs_name)
#         ll_array = np.ones_like(hue_array) * ll
#         plane_lut = tstack([ll_array, chroma_array, hue_array])
#         lut[l_idx] = plane_lut
#         mtime.lap()

#     mtime.end()
#     # print(lut)

#     return lut


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
        # print(f"l_idx={l_idx}, addr={st_addr}, lut={plane_lut[l_idx]}")
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
    # create buffer (2D-LUT)
    # lut = np.zeros((lightness_sample, hue_sample, 3))

    total_process_num = hue_sample
    block_process_num = cpu_count()
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
        mtime.lap()
    mtime.end()

    lut = np.array(
        shared_array[:lightness_sample*hue_sample*3]).reshape(
            (lightness_sample, hue_sample, 3))

    # mtime = MeasureExecTime()
    # mtime.start()
    # for h_idx in range(hue_sample):
    #     hh = h_idx / (hue_sample - 1) * 360
    #     print(f"h_idx={h_idx}, h_val={hh}")
    #     lightness_array, chroma_array = calc_chroma_boundary_specific_hue(
    #         hue=hh, chroma_sample=chroma_sample,
    #         ll_num=lightness_sample, cs_name=cs_name)
    #     hue_array = np.ones_like(lightness_array) * hh
    #     plane_lut = tstack([lightness_array, chroma_array, hue_array])
    #     lut[:, h_idx] = plane_lut
    #     mtime.lap()

    # mtime.end()
    # # print(lut)

    return lut


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # # calc_chroma_boundary_specific_l(ll=50, cs_name=cs.BT2020)
    # chroma_sample = 32768
    # lightness_sample = 1024
    # lut = calc_chroma_boundary_lut(
    #     lightness_sample=50, chroma_sample=8192, chroma_max=220,
    #     hue_num=361, cs_name=cs.BT2020)
    # np.save("./lut_sample_50_361_8192.npy", lut)

    hue = 0
    hue_sample = 1024
    chroma_sample = 16384
    ll_num = 1024
    cs_name = cs.BT2020
    # calc_chroma_boundary_specific_hue(
    #     hue=hue, chroma_sample=chroma_sample,
    #     ll_num=ll_num, cs_name=cs_name)
    lut = calc_chroma_boundary_lut(
        lightness_sample=ll_num, chroma_sample=chroma_sample,
        hue_sample=hue_sample, cs_name=cs_name)
    np.save(
        f"./lut/lut_sample_{ll_num}_{hue_sample}_{chroma_sample}.npy", lut)
