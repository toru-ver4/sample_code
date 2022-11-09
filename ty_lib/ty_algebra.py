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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    pos1 = [-4, -1]
    pos2 = [2, 2]
    a, b = calc_linear_function_params(pos1=pos1, pos2=pos2)
    print(a, b)
