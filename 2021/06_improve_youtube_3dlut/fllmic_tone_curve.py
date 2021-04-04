# -*- coding: utf-8 -*-
"""
This source code is copied from https://www.desmos.com/calculator/12vlon6rpu?lang=ja 
And original article is here http://filmicworlds.com/blog/filmic-tonemapping-with-piecewise-power-curves/ 
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = '-'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def curve_segment(x, m_lnA, m_B, m_offsetX, m_offsetY, m_scaleX=1, m_scaleY=1):
    x0 = (x - m_offsetX) * m_scaleX
    y0 = np.exp(m_lnA + m_B * np.log(x0))
    return y0 * m_scaleY + m_offsetY


def solve_A_B(x0, y0, m):
    B = (m * x0) / y0
    lnA = np.log(y0) - B * np.log(x0)
    return lnA, B


def as_slope_intercept(x0, x1, y0, y1):
    dx = (x1 - x0)
    dy = (y1 - y0)

    m = 1 if dx == 0 else dy / dx
    b = y0 - x0 * m

    return m, b


def eval_derivative_linear_gamma(m, b, g, x):
    return g * m * np.power(m * x + b, g - 1.0)


def curve(x, g, m_x0, m_y0, m_x1, m_y1, m_overshoot_x=0, m_overshoot_y=0):
    m, b = as_slope_intercept(m_x0, m_x1, m_y0, m_y1)

    linear_segment = curve_segment(x, g * np.log(m), g, -(b / m), 0)

    toe_M = eval_derivative_linear_gamma(m, b, g, m_x0)
    shoulder_M = eval_derivative_linear_gamma(m, b, g, m_x1)

    m_y0 = np.power(m_y0, g)
    m_y1 = np.power(m_y1, g)
    m_overshoot_y = np.power(1 + m_overshoot_y, g) - 1

    A_t, B_t = solve_A_B(m_x0, m_y0, toe_M)
    toe_segment = curve_segment(x, A_t, B_t, 0, 0)

    x0 = (1 + m_overshoot_x) - m_x1
    y0 = (1 + m_overshoot_y) - m_y1
    A_s, B_s = solve_A_B(x0, y0, shoulder_M)
    shoulder_segment = curve_segment(x, A_s, B_s, 1 + m_overshoot_x,
                                     1 + m_overshoot_y, -1, -1)

    return np.select(
        [x < m_x0, np.logical_and(x > m_x0, x < m_x1), x > m_x1],
        [toe_segment, linear_segment, shoulder_segment])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    samples = np.linspace(0, 1, 100)

    m_x0, m_y0, m_x1, m_y1 = 0.25, 0.15, 0.75, 0.85
    plt.plot(samples, curve(samples, 1, m_x0, m_y0, m_x1, m_y1))
    plt.show()
