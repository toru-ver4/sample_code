# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_linear_function_params(pos1, pos2):
    """
    Calculate parameter `a` and `b` of `y=ax+b`.

    Parameters
    ----------
    pos1 : list
        [x1, y1]
    pos2 : list
        [x2, y2]

    Returns
    -------
    list
        [a, b]

    Examples
    --------
    >>> pos1 = [-4, -1]
    >>> pos2 = [2, 2]
    >>> a, b = calc_linear_function_params(pos1=pos1, pos2=pos2)
    >>> print(a, b)
    0.5 1.0
    """
    x1, y1 = pos1
    x2, y2 = pos2
    a = (y2 - y1)/(x2 - x1)
    b = -a * x1 + y1

    return a, b


def calc_y_from_three_pos(x, pos1, pos2, pos3):
    """
    Examples
    --------
    """
    a1, b1 = calc_linear_function_params(pos1=pos1, pos2=pos2)
    a2, b2 = calc_linear_function_params(pos1=pos2, pos2=pos3)
    x2, _ = pos2

    y = np.where(
        x < x2,
        a1 * x + b1,
        a2 * x + b2)

    return y


def calc_y_from_four_pos(x, pos1, pos2, pos3, pos4):
    """
    Examples
    --------
    """
    a1, b1 = calc_linear_function_params(pos1=pos1, pos2=pos2)
    a2, b2 = calc_linear_function_params(pos1=pos2, pos2=pos3)
    a3, b3 = calc_linear_function_params(pos1=pos3, pos2=pos4)
    x2, _ = pos2
    x3, _ = pos3

    y = np.zeros_like(x)
    line1_idx = x < x2
    line2_idx = (x >= x2) & (x < x3)
    line3_idx = (x >= x3)

    y[line1_idx] = a1 * x[line1_idx] + b1
    y[line2_idx] = a2 * x[line2_idx] + b2
    y[line3_idx] = a3 * x[line3_idx] + b3

    return y


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    pos1 = [-4, -1]
    pos2 = [2, 2]
    pos3 = [5, 8]
    pos4 = [9, 0]
    a, b = calc_linear_function_params(pos1=pos1, pos2=pos2)
    print(a, b)

    x = np.linspace(pos1[0], pos4[0], 64)
    y = calc_y_from_four_pos(x, pos1, pos2, pos3, pos4)

    import plot_utility as pu
    fig, ax1 = pu.plot_1_graph()
    ax1.plot(x, y)
    pu.show_and_save(fig=fig, save_fname="aaa.png")
