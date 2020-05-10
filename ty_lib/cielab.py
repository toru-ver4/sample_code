# -*- coding: utf-8 -*-
"""
CIELAB の特殊操作用
=====================

"""

# import standard libraries
import os

# import third-party libraries
# import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
from sympy import sin, cos
from sympy.solvers import solve
from scipy import linalg
from colour import xy_to_XYZ, Lab_to_XYZ, XYZ_to_RGB
from colour import RGB_COLOURSPACES
from colour.models import BT709_COLOURSPACE

# import my libraries
import color_space as cs

# definition
D65_WHITE = xy_to_XYZ(cs.D65) * 100
D65_X = D65_WHITE[0]
D65_Y = D65_WHITE[1]
D65_Z = D65_WHITE[2]

SIGMA = 6/29

IJK_LIST = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]]

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

im_threshold = 0.00000000001


def get_ty(l):
    """
    l, c, h は Sympy の Symbol
    """
    return (l + 16) / 116


def get_tx(l, c, h):
    """
    l, c, h は Sympy の Symbol
    """
    return get_ty(l) + (c * cos(h))/500


def get_tz(l, c, h):
    """
    l, c, h は Sympy の Symbol
    """
    return get_ty(l) - (c * sin(h))/200


def get_large_x(l, c, h, ii):
    """
    l, c, h は Sympy の Symbol.
    ii==0 は t <= sigma, ii==1 は t > sigma のパターン。
    """
    if ii == 0:
        ret_val = D65_X / 100 * 3 * (SIGMA ** 2) * (get_tx(l, c, h) - 4 / 29)
    else:
        ret_val = D65_X / 100 * (get_tx(l, c, h) ** 3)

    return ret_val


def get_large_y(l, c, h, jj):
    """
    l, c, h は Sympy の Symbol.
    jj==0 は t <= sigma, jj==1 は t > sigma のパターン。
    """
    if jj == 0:
        ret_val = D65_Y / 100 * 3 * (SIGMA ** 2) * (get_ty(l) - 4 / 29)
    else:
        ret_val = D65_Y / 100 * (get_ty(l) ** 3)

    return ret_val


def get_large_z(l, c, h, kk):
    """
    l, c, h は Sympy の Symbol.
    kk==0 は t <= sigma, kk==1 は t > sigma のパターン。
    """
    if kk == 0:
        ret_val = D65_Z / 100 * 3 * (SIGMA ** 2) * (get_tz(l, c, h) - 4 / 29)
    else:
        ret_val = D65_Z / 100 * (get_tz(l, c, h) ** 3)

    return ret_val


def get_xyz_to_rgb_matrix(primaries=cs.get_primaries(cs.BT2020)):
    rgb_to_xyz_matrix = cs.calc_rgb_to_xyz_matrix(
        gamut_xy=primaries, white_large_xyz=D65_WHITE)
    xyz_to_rgb_matrix = linalg.inv(rgb_to_xyz_matrix)
    return xyz_to_rgb_matrix


def lab_to_rgb_expr(l, c, h, primaries=cs.get_primaries(cs.BT2020)):
    mtx = get_xyz_to_rgb_matrix(primaries=primaries)
    ret_list = []
    for ijk in IJK_LIST:
        r = mtx[0][0] * get_large_x(l, c, h, ijk[0]) + mtx[0][1] * get_large_y(l, c, h, ijk[1]) + mtx[0][2] * get_large_z(l, c, h, ijk[2])
        g = mtx[1][0] * get_large_x(l, c, h, ijk[0]) + mtx[1][1] * get_large_y(l, c, h, ijk[1]) + mtx[1][2] * get_large_z(l, c, h, ijk[2])
        b = mtx[2][0] * get_large_x(l, c, h, ijk[0]) + mtx[2][1] * get_large_y(l, c, h, ijk[1]) + mtx[2][2] * get_large_z(l, c, h, ijk[2])
        ret_list.append([r, g, b])

    return ret_list


def solve_chroma(
        l_val, l_idx, h_val, h_idx, rgb_exprs, l, c, h, l_sample_num, **kwarg):
    """
    引数で与えられた L*, H に対する Chroma値を算出する。
    ```make_chroma_array``` のループからコールされることが前提のコード。
    """
    if l_sample_num == 0:  # 特定の ab平面計算用の引数
        pass
    elif l_idx == l_sample_num - 1:  # L=100 の Chroma は 0 なので計算しない
        return 0
    elif l_idx == 0:   # L=0 の Chroma は 0 なので計算しない
        return 0
    # start = time.time()
    xyz_t = [get_tx(l, c, h), get_ty(l), get_tz(l, c, h)]
    xyz_t = [xyz_t[idx].subs({l: l_val, h: h_val}) for idx in range(3)]
    temp_solution = []

    for ii in range(len(IJK_LIST)):
        for jj in range(3):  # R, G, B のループ
            # l_val, h_val 代入
            c_expr = rgb_exprs[ii][jj].subs({l: l_val, h: h_val})
            solution = []
            solution.extend(solve(c_expr))
            solution.extend(solve(c_expr - 1))

            for solve_val_complex in solution:
                # 複素成分を見て、小さければ実数とみなす
                # どうも solve では複素数として算出されてしまうケースがあるっぽい
                solve_val, im_val = solve_val_complex.as_real_imag()
                if abs(im_val) > im_threshold:
                    continue

                t = [xyz_t[kk].subs({c: solve_val}) for kk in range(3)]
                xt_bool = (t[0] > SIGMA) if IJK_LIST[ii][0] else (t[0] <= SIGMA)
                yt_bool = (t[1] > SIGMA) if IJK_LIST[ii][1] else (t[1] <= SIGMA)
                zt_bool = (t[2] > SIGMA) if IJK_LIST[ii][2] else (t[2] <= SIGMA)
                xyz_t_bool = (xt_bool and yt_bool) and zt_bool
                if xyz_t_bool:
                    temp_solution.append(solve_val)

    chroma_list = np.array(temp_solution)
    chroma = np.min(chroma_list[chroma_list >= 0.0])

    print("L*={:.2f}, H={:.2f}, C={:.3f}".format(
            l_val, h_val / (2 * np.pi) * 360, chroma))
    return chroma


def is_inner_gamut(lab, color_space_name=cs.BT709):
    rgb = XYZ_to_RGB(
        Lab_to_XYZ(lab), cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].XYZ_to_RGB_matrix)
    r_judge = (rgb[..., 0] >= 0) & (rgb[..., 0] <= 1)
    g_judge = (rgb[..., 1] >= 0) & (rgb[..., 1] <= 1)
    b_judge = (rgb[..., 2] >= 0) & (rgb[..., 2] <= 1)
    judge = (r_judge & g_judge) & b_judge

    return judge


def solve_chroma_fast(
        l_val, l_idx, h_val, h_idx, l_sample_num, color_space_name, **kwarg):
    """
    引数で与えられた L*, H に対する Chroma値を算出する。
    ```make_chroma_array``` のループからコールされることが前提のコード。
    """
    if l_sample_num == 0:  # 特定の ab平面計算用の引数
        pass
    elif l_idx == l_sample_num - 1:  # L=100 の Chroma は 0 なので計算しない
        return 0
    elif l_idx == 0:   # L=0 の Chroma は 0 なので計算しない
        return 0

    r_val_init = 300
    trial_num = 50

    r_val = r_val_init

    for t_idx in range(trial_num):
        # print(f"t_idx={t_idx}, r_val={r_val}")
        aa = r_val * np.cos(h_val)
        bb = r_val * np.sin(h_val)
        lab = np.array([l_val, aa, bb])
        judge = is_inner_gamut(lab=lab, color_space_name=color_space_name)
        # print(f"judge={judge}")
        add_sub = r_val_init / (2 ** (t_idx + 1))
        if judge:
            r_val = r_val + add_sub
        else:
            r_val = r_val - add_sub

    chroma = r_val
    print("L*={:.2f}, H={:.2f}, C={:.3f}".format(
            l_val, np.rad2deg(h_val), chroma))
    return chroma


def solve_chroma_wrapper(args):
    solve_chroma(*args)


def _calc_bilinear_sample_data(lh, l_sample_num, h_sample_num):
    """
    CIELAB空間の特定の色域の Gamut Boundary に対して
    Bilinear Interpolation を行うためのサンプル点の抽出を行う。

    Parameters
    ----------
    lh : array_like
        L* and Hue data.
    l_sample_num : int
        Number of samples in Lihgtness direction.
    h_sample_num : int
        Number of samples in Hue direction.

    Returns
    -------
    array_like
        Four sample indeces used for Bilinear Interpolation.
        And two ratio data for interpolation.

    Examples
    --------
    >>> test_data = np.array(
    ...     [[0.0, 0.0], [100.0, 2 * np.pi],
    ...      [0.00005, 0.00001 * np.pi], [99.99995, 1.99999 * np.pi]])
    >>> indices, ratios = _calc_bilinear_sample_data(
    ...     lh=test_data, l_sample_num=256, h_sample_num=256)
    >>> print(indices)
    >>> [[[  0.   0.   0.   0.]
    ...   [255. 255. 255. 255.]
    ...   [  1.   0.   1.   0.]
    ...   [255. 254. 255. 254.]]]
    >>> print(ratios)
    >>> [[[0.000000e+00, 0.000000e+00],
    ...   [1.275000e-04, 1.275000e-03],
    ...   [9.998725e-01, 9.987250e-01]]]
    """
    l_temp = lh[..., 0] / 100 * (l_sample_num - 1)
    h_temp = lh[..., 1] / (2 * np.pi) * (h_sample_num - 1)
    l_hi = np.uint16(np.ceil(l_temp))
    l_lo = np.uint16(np.floor(l_temp))
    h_hi = np.uint16(np.ceil(h_temp))
    h_lo = np.uint16(np.floor(h_temp))
    r_l = l_hi - l_temp  # ratio in Luminance direction
    r_h = h_hi - h_temp  # ratio in Hue direction

    return np.dstack((l_hi, l_lo, h_hi, h_lo)), np.dstack((r_l, r_h))


def bilinear_interpolation(lh, lut2d):
    """
    Bilinear で補間します。

    Parameters
    ----------
    lh : array_like
        L* and Hue data.
    lut2d : array_like
        2d lut data.

    Returns
    -------
    array_like
        Chroma value calculated from four sample indices
        using Bilinear Interpolation.

    Examples
    --------
    >>> # 256x256 は密集すぎなのでスカスカな LUT を作成
    >>> chroma_lut = np.load("./boundary_data/Chroma_BT709_l_256_h_256.npy")
    >>> h_idx = np.arange(0, 256, 8)
    >>> l_idx = np.arange(0, 256, 64)
    >>> sparse_lut = chroma_lut[l_idx][:, h_idx]
    >>>
    >>> # 補間の入力データ lh を作成
    >>> l_val = 40
    >>> target_h_num = 256
    >>> ll = l_val * np.ones(target_h_num)
    >>> hue = np.linspace(0, 2 * np.pi, target_h_num)
    >>> lh = np.dstack((ll, hue))
    >>>
    >>> # 補間実行
    >>> chroma_interpolation = bilinear_interpolation(lh, sparse_lut)
    """
    l_sample_num = lut2d.shape[0]
    h_sample_num = lut2d.shape[1]
    indices, ratios = _calc_bilinear_sample_data(
        lh, l_sample_num, h_sample_num)
    l_hi_idx = indices[..., 0]
    l_lo_idx = indices[..., 1]
    h_hi_idx = indices[..., 2]
    h_lo_idx = indices[..., 3]
    l_ratio = ratios[..., 0]
    h_ratio = ratios[..., 1]

    # interpolation in Hue direction
    temp_hi = lut2d[l_hi_idx, h_hi_idx] * (1 - h_ratio)\
        + lut2d[l_hi_idx, h_lo_idx] * h_ratio
    temp_lo = lut2d[l_lo_idx, h_hi_idx] * (1 - h_ratio)\
        + lut2d[l_lo_idx, h_lo_idx] * h_ratio

    # interpolation in Luminance direction
    result = temp_hi * (1 - l_ratio) + temp_lo * l_ratio

    return result[0]


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    l_sample = 64
    h_sample = 64
    hue_list = np.deg2rad(np.linspace(0, 360, h_sample))
    lightness_list = np.linspace(0, 100, l_sample)
    for l_idx, l_val in enumerate(lightness_list):
        for h_idx, h_val in enumerate(hue_list):
            solve_chroma2(
                l_val=l_val, l_idx=l_idx, h_val=h_val, h_idx=h_idx,
                l_sample_num=l_sample, color_space_name=cs.BT709)

