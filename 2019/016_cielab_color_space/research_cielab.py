# -*- coding: utf-8 -*-
"""
CIELAB色空間の基礎調査
=====================

* XYZ色空間と Lab色空間の順変換・逆変換の数式を確認
* CLELAB a*b* plane (以後は a*b* plane と略す) のプロット(L を 0～100 まで 0.1 step で)
* CIELAB C*L* plane (以後は C*L* plane と略す) のプロット(h を 0～360 まで 0.5 step で)

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from sympy import symbols, solve, plotting, sin, cos
from scipy import linalg
from colour.models import BT2020_COLOURSPACE, BT709_COLOURSPACE
from colour import xy_to_XYZ

# import my libraries
import color_space as cs

# definition
D65_X = 95.04
D65_Y = 100.0
D65_Z = 108.89
D65_WHITE = [D65_X, D65_Y, D65_Z]

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def check_basic_trigonometricfunction():
    print(np.sin(np.pi * -4 / 4))
    print(np.sin(np.pi * -2 / 4))
    print(np.sin(np.pi * 2 / 4))
    print(np.sin(np.pi * 4 / 4))


def get_inv_f_upper():
    """
    t > sigma の f^-1 を返す
    """
    t = symbols('t')
    return t ** 3, t


def get_inv_f_lower():
    """
    t <= sigma の f^-1 を返す
    """
    t = symbols('t')
    sigma = 6/29
    return 3 * (sigma ** 2) * (t - 4 / 29), t


def plot_inv_f():
    upper, t = get_inv_f_upper()
    plotting.plot(upper, (t, -1, 1))

    lower, t = get_inv_f_lower()
    plotting.plot(lower, (t, -1, 1))


def get_large_xyz_symbol(n, t, upper=True):
    """
    example
    -------
    c, l, h = symbols('c, l, h')
    xt = (l + 16) / 116 + (c * cos(h)) / 500
    yt = (l + 16) / 116
    zt = (l + 16) / 116 - (c * sin(h)) / 200

    x = get_large_xyz_symbol(n=D65_X, t=xt, upper=True)
    y = get_large_xyz_symbol(n=D65_Y, t=yt, upper=True)
    z = get_large_xyz_symbol(n=D65_Z, t=zt, upper=True)
    """
    func, u = get_inv_f_upper() if upper else get_inv_f_lower()
    print(func)
    return n / 100 * func.subs({u: t})


def apply_matrix(src, mtx):
    """
    src: [3]
    mtx: [3][3]
    """
    a = src[0] * mtx[0][0] + src[1] * mtx[0][1] + src[2] * mtx[0][2]
    b = src[0] * mtx[1][0] + src[1] * mtx[1][1] + src[2] * mtx[1][2]
    c = src[0] * mtx[2][0] + src[1] * mtx[2][1] + src[2] * mtx[2][2]

    return a, b, c


def get_xyz_to_rgb_matrix(primaries=cs.REC2020_xy):
    rgb_to_xyz_matrix = cs.calc_rgb_to_xyz_matrix(
        gamut_xy=primaries, white_large_xyz=D65_WHITE)
    xyz_to_rgb_matrix = linalg.inv(rgb_to_xyz_matrix)
    return xyz_to_rgb_matrix


def lab_to_xyz_formla():
    c, l, h = symbols('c, l, h')
    xt = (l + 16) / 116 + (c * cos(h)) / 500
    yt = (l + 16) / 116
    zt = (l + 16) / 116 - (c * sin(h)) / 200

    matrix = get_xyz_to_rgb_matrix(primaries=cs.REC2020_xy)

    # upper
    xu = get_large_xyz_symbol(n=D65_X, t=xt, upper=True)
    yu = get_large_xyz_symbol(n=D65_Y, t=yt, upper=True)
    zu = get_large_xyz_symbol(n=D65_Z, t=zt, upper=True)
    ru, gu, bu = apply_matrix([xu, yu, zu], matrix)

    # lower
    xd = get_large_xyz_symbol(n=D65_X, t=xt, upper=False)
    yd = get_large_xyz_symbol(n=D65_Y, t=yt, upper=False)
    zd = get_large_xyz_symbol(n=D65_Z, t=zt, upper=False)
    rd, gd, bd = apply_matrix([xd, yd, zd], matrix)

    l_val = 50
    h_val = np.pi / 4

    ru = ru.subs({l: l_val, h: h_val})
    gu = gu.subs({l: l_val, h: h_val})
    bu = bu.subs({l: l_val, h: h_val})

    # plotting.plot(ru, (c, -200, 200))
    # plotting.plot(gu, (c, -200, 200))
    # plotting.plot(bu, (c, -200, 200))

    rd = rd.subs({l: l_val, h: h_val})
    gd = gd.subs({l: l_val, h: h_val})
    bd = bd.subs({l: l_val, h: h_val})

    plotting.plot(rd, (c, -200, 200))
    plotting.plot(gd, (c, -200, 200))
    plotting.plot(bd, (c, -200, 200))


def experimental_functions():
    # check_basic_trigonometricfunction()
    # plot_inv_f()
    lab_to_xyz_formla()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_functions()
