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
import time
import ctypes

# import third-party libraries
# import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
from sympy import symbols, plotting, sin, cos, lambdify, pi, roots
from sympy.solvers import solve
from sympy.solvers.solveset import solveset, solvify

# definition
D65_X = 95.04
D65_Y = 100.0
D65_Z = 108.89
D65_WHITE = [D65_X, D65_Y, D65_Z]
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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    c = symbols('c')
    # expr = -0.0006587933118851*c + 0.0167674289909477*(1.22464679914735e-19*c + 0.425287356321839)**3 + 0.0345712150974614
    expr = -0.0006587933118851*c*sin(8*pi/21) + 0.0167674289909477*(c*cos(8*pi/21)/500 + 0.425287356321839)**3 + 0.0345712150974614
    print(expr)
    start = time.time()
    solution = solve(expr)
    end = time.time()
    print(solution)
    print("total_time={}[s]".format(end-start))

    # for sol_val in solution:
    #     print((sol_val.as_real_imag()[1]))

    # expr2 = c ** 2 - 1
    # solution = solve(expr2)
    for sol_val in solution:
        print((sol_val.as_real_imag()[1].evalf()))
    # end = time.time()
    # print("total_time={}[s]".format(end-start))
    print(10 ** (-10))