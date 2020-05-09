# -*- coding: utf-8 -*-
"""
BT2407 実装用の各種LUTを作成する
===============================

"""

# import standard libraries
import os
import ctypes
import time

# import third-party libraries
from sympy import symbols
import numpy as np
from multiprocessing import Pool, cpu_count, Array


# import my libraries
import cielab as cl
import color_space as cs
from bt2407_parameters import L_SAMPLE_NUM_MAX, H_SAMPLE_NUM_MAX,\
    GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE, GAMUT_BOUNDARY_LUT_HUE_SAMPLE,\
    get_gamut_boundary_lut_name

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


shared_array = Array(
    typecode_or_type=ctypes.c_float,
    size_or_initializer=L_SAMPLE_NUM_MAX*H_SAMPLE_NUM_MAX)


def solve_chroma_wrapper(args):
    chroma = cl.solve_chroma(**args)
    s_idx = args['h_sample_num'] * args['l_idx'] + args['h_idx']
    shared_array[s_idx] = chroma


def make_chroma_array(primaries=cs.get_primaries(cs.BT709),
                      l_sample_num=L_SAMPLE_NUM_MAX,
                      h_sample_num=H_SAMPLE_NUM_MAX):
    """
    L*a*b* 空間における a*b*平面の境界線プロットのために、
    各L* における 境界線の Chroma を計算する。
    """

    l, c, h = symbols('l, c, h')
    rgb_exprs = cl.lab_to_rgb_expr(l, c, h, primaries=primaries)
    l_vals = np.linspace(0, 100, l_sample_num)
    h_vals = np.linspace(0, 2*np.pi, h_sample_num)
    for l_idx, l_val in enumerate(l_vals):
        args = []
        for h_idx, h_val in enumerate(h_vals):
            d = dict(
                l_val=l_val, l_idx=l_idx, h_val=h_val, h_idx=h_idx,
                rgb_exprs=rgb_exprs, l=l, c=c, h=h,
                l_sample_num=l_sample_num, h_sample_num=h_sample_num)
            args.append(d)
        with Pool(cpu_count()) as pool:
            pool.map(solve_chroma_wrapper, args)

    chroma = np.array(
        shared_array[:l_sample_num * h_sample_num]).reshape(
            (l_sample_num, h_sample_num))
    return chroma


def make_gamut_bondary_lut(
        l_sample_num=GAMUT_BOUNDARY_LUT_LUMINANCE_SAMPLE,
        h_sample_num=GAMUT_BOUNDARY_LUT_HUE_SAMPLE,
        color_space_name=cs.BT709):
    chroma = make_chroma_array(
        primaries=cs.get_primaries(color_space_name),
        l_sample_num=l_sample_num, h_sample_num=h_sample_num)
    fname = get_gamut_boundary_lut_name(
        color_space_name, l_sample_num, h_sample_num)
    np.save(fname, chroma)


def make_gamut_boundary_lut_all():
    # L*a*b* 全体のデータを算出
    start = time.time()
    make_gamut_bondary_lut(color_space_name=cs.BT709)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    make_gamut_bondary_lut(color_space_name=cs.BT2020)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    make_gamut_bondary_lut(color_space_name=cs.P3_D65)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


def main_func():
    # make_gamut_boundary_lut_all()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
