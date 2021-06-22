# -*- coding: utf-8 -*-
"""
create gamut boundary lut of Jzazbz color space.
"""

# import standard libraries
import os
import ctypes
from colour.utilities.array import tstack

# import third-party libraries
import numpy as np
from multiprocessing import Pool, cpu_count, Array
from scipy import signal
from colour import xy_to_XYZ

# import my libraries
import color_space as cs
from create_gamut_booundary_lut\
    import calc_chroma_boundary_specific_ligheness_jzazbz,\
    create_focal_point_lut_jzazbz,\
    calc_chroma_boundary_specific_ligheness_jzazbz_type2,\
    get_gamut_boundary_lch_from_lut
from jzazbz import large_xyz_to_jzazbz
from common import MeasureExecTime

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

shared_array_type = ctypes.c_float
shared_array_elem_size = ctypes.sizeof(shared_array_type)
shared_array = Array(
    typecode_or_type=shared_array_type,
    size_or_initializer=L_SAMPLE_NUM_MAX*H_SAMPLE_NUM_MAX*COLOR_NUM)


class TyLchLut():
    """
    Toru Yoshihara's Lightness-Chroma-Hue Lut.
    """
    def __init__(self, lut):
        self.lut = lut
        self.jz_min = lut[0, 0, 0]
        self.jz_max = lut[-1, 0, 0]
        print(f"min, max = {self.jz_min}, {self.jz_max}")

    def interpolate(self, lh_array):
        return get_gamut_boundary_lch_from_lut(
            lut=self.lut, lh_array=lh_array, lightness_max=self.jz_max)


def make_gb_lut_fname(
        color_space_name, luminance, lightness_num, hue_num):
    fname = f"./lut/JzChz_gb-lut_type2_{color_space_name}_"
    fname += f"{luminance}nits_jj-{lightness_num}_"
    fname += f"hh-{hue_num}.npy"

    return fname


def make_focal_lut_fname(
        luminance, lightness_num, hue_num, prefix="BT709_BT2020"):
    fname = f"./lut/JzChz_focal-lut_{prefix}_"
    fname += f"{luminance}nits_jj-{lightness_num}_"
    fname += f"hh-{hue_num}.npy"

    return fname


def make_focal_lut_fname_wo_lpf(
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

    fname = make_gb_lut_fname(
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
    block_process_num = 4  # for 32768 sample
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

    fname = make_gb_lut_fname(
        color_space_name=color_space_name, luminance=luminance,
        lightness_num=lightness_sample, hue_num=hue_sample)
    np.save(fname, np.float32(lut))


def low_pass_filter2(x, nn=4, wn=0.25):
    b1, a1 = signal.bessel(nn, wn, "low")
    result = signal.filtfilt(b1, a1, x)

    return result


def apply_lpf_to_focal_lut(
        luminance, lightness_num, hue_num, prefix="BT709-BT2020",
        maximum_l_focal=1.0, minimum_l_focal=0.0):
    inner_lut_name = make_gb_lut_fname(
        color_space_name=cs.BT709, luminance=luminance,
        lightness_num=lightness_num, hue_num=hue_num)
    outer_lut_name = make_gb_lut_fname(
        color_space_name=cs.BT2020, luminance=luminance,
        lightness_num=lightness_num, hue_num=hue_num)
    inner_lut = TyLchLut(lut=np.load(inner_lut_name))
    outer_lut = TyLchLut(lut=np.load(outer_lut_name))

    focal_array_wo_lpf = create_focal_point_lut_jzazbz(
        inner_lut=inner_lut, outer_lut=outer_lut,
        maximum_l_focal=maximum_l_focal, minimum_l_focal=minimum_l_focal)

    ll = focal_array_wo_lpf[..., 0]
    ll_new = low_pass_filter2(
        ll, nn=4, wn=0.25)

    focal_array_w_lpf = tstack(
        [ll_new, focal_array_wo_lpf[..., 1], focal_array_wo_lpf[..., 2]])

    focal_lut_w_lpf_name = make_focal_lut_fname(
        luminance=luminance,
        lightness_num=lightness_sample_num, hue_num=hue_sample_num,
        prefix=prefix)
    focal_lut_wo_lpf_name = make_focal_lut_fname_wo_lpf(
        luminance=luminance,
        lightness_num=lightness_sample_num, hue_num=hue_sample_num,
        prefix=prefix)
    np.save(focal_lut_w_lpf_name, focal_array_w_lpf)
    np.save(focal_lut_wo_lpf_name, focal_array_wo_lpf)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=256, lightness_sample=256,
    #     color_space_name=cs.BT2020, luminance=10000)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=16, lightness_sample=256,
    #     color_space_name=cs.BT2020, luminance=10000)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=16, lightness_sample=256,
    #     color_space_name=cs.BT709, luminance=10000)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=64, lightness_sample=64,
    #     color_space_name=cs.BT2020, luminance=10000)
    # create_jzazbz_gamut_boundary_lut(
    #     hue_sample=64, lightness_sample=64,
    #     color_space_name=cs.BT709, luminance=10000)
    lightness_sample_num = L_SAMPLE_NUM_MAX
    hue_sample_num = H_SAMPLE_NUM_MAX
    luminance = 10000
    chroma_sample = 16384
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT709,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT2020,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.P3_D65,
    #     luminance=luminance)

    # luminance = 1000
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT709,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT2020,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.P3_D65,
    #     luminance=luminance)

    # luminance = 100
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT709,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.BT2020,
    #     luminance=luminance)
    # create_jzazbz_gamut_boundary_lut_type2(
    #     hue_sample=hue_sample_num, lightness_sample=lightness_sample_num,
    #     chroma_sample=chroma_sample, color_space_name=cs.P3_D65,
    #     luminance=luminance)

    # luminance = 10000
    # apply_lpf_to_focal_lut(
    #     luminance, lightness_sample_num, hue_sample_num,
    #     prefix="BT709-BT2020",
    #     maximum_l_focal=1.0, minimum_l_focal=0.0)
    # luminance = 1000
    # apply_lpf_to_focal_lut(
    #     luminance, lightness_sample_num, hue_sample_num,
    #     prefix="BT709-BT2020",
    #     maximum_l_focal=1.0, minimum_l_focal=0.0)
    luminance = 100
    apply_lpf_to_focal_lut(
        luminance, lightness_sample_num, hue_sample_num,
        prefix="BT709-BT2020",
        maximum_l_focal=0.132, minimum_l_focal=0.06)

    # lut_name = make_gb_lut_fname(
    #     color_space_name=cs.BT709, luminance=luminance,
    #     lightness_num=lightness_sample_num, hue_num=hue_sample_num)
    # lut = TyLchLut(lut=np.load(lut_name))
