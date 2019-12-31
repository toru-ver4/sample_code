# -*- coding: utf-8 -*-
"""
solve で虚数解が出てしまう件の調査
=================================
"""

# import standard libraries
import os

# import third-party libraries
import matplotlib.pyplot as plt

import numpy as np
from sympy import symbols
from sympy.solvers import solve

# import my libraries
import plot_utility as pu

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


def plot_y(val_range=(-20000, 20000)):
    x = np.linspace(val_range[0], val_range[1], 1024)
    a = 6.54 * (10 ** -12)
    b = 1.14 * (10 ** -11)
    c = -6.07 * (10 ** -4)
    d = 0.0359
    y = a * (x ** 3) + b * (x ** 2) + c * x + d
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="3次関数のプロット",
        graph_title_size=None,
        xlabel="x value",
        ylabel="y value",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)

    ax1.plot(x, y)
    fname = "range_{}_{}.png".format(val_range[0], val_range[1])
    plt.savefig("./blog_img/" + fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def solve_cubic_function():
    x = symbols('x')
    a = 6.54 * (10 ** -12)
    b = 1.14 * (10 ** -11)
    c = -6.07 * (10 ** -4)
    d = 0.0359
    expr = a * (x ** 3) + b * (x ** 2) + c * x + d
    solution = solve(expr)
    print(solution)


def solve_cubic_function2():
    threshold = 10 ** -10
    x = symbols('x')
    a = 6.54 * (10 ** -12)
    b = 1.14 * (10 ** -11)
    c = -6.07 * (10 ** -4)
    d = 0.0359
    expr = a * (x ** 3) + b * (x ** 2) + c * x + d
    solutions = solve(expr)

    real_solutions = []
    for value in solutions:
        real_val, im_val = value.as_real_imag()
        if abs(im_val) < threshold:
            real_solutions.append(real_val)

    print(real_solutions)


def solve_cubic_function3():
    x = symbols('x', real=True)
    a = 6.54 * (10 ** -12)
    b = 1.14 * (10 ** -11)
    c = -6.07 * (10 ** -4)
    d = 0.0359
    expr = a * (x ** 3) + b * (x ** 2) + c * x + d
    solution = solve(expr)
    print(solution)


def output_solve_expr():
    a, b, c, d, x = symbols('a, b, c, d, x')
    expr = a * (x ** 3) + b * (x ** 2) + c * x + d
    print(solve(expr, x))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_y(val_range=(-20000, 20000))
    plot_y(val_range=(-50, 150))
    solve_cubic_function()
    solve_cubic_function2()
    solve_cubic_function3()
    # c = symbols('c')
    # expr = -0.0006587933118851*c + 0.0167674289909477*(1.22464679914735e-19*c + 0.425287356321839)**3 + 0.0345712150974614
    # expr = -0.0006587933118851*c*sin(8*pi/21) + 0.0167674289909477*(c*cos(8*pi/21)/500 + 0.425287356321839)**3 + 0.0345712150974614
    # print(expr.evalf().expand())
    # start = time.time()
    # solution = solve(expr)
    # end = time.time()
    # print(solution)
    # print("total_time={}[s]".format(end-start))

    # # for sol_val in solution:
    # #     print((sol_val.as_real_imag()[1]))

    # # expr2 = c ** 2 - 1
    # # solution = solve(expr2)
    # for sol_val in solution:
    #     print((sol_val.as_real_imag()[1].evalf()))
    # # end = time.time()
    # # print("total_time={}[s]".format(end-start))
    # print(10 ** (-10))
