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
from colour import xy_to_XYZ

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
    if l_idx == l_sample_num - 1:  # L=100 の Chroma は 0 なので計算しない
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

    # s_idx = h_sample_num * l_idx + h_idx
    # shared_array[s_idx] = chroma
    print("L*={:.2f}, H={:.2f}, C={:.3f}".format(
            l_val, h_val / (2 * np.pi) * 360, chroma))
    # end = time.time()
    # print("each_time={}[s]".format(end-start))
    return chroma


def solve_chroma_wrapper(args):
    solve_chroma(*args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
