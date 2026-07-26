# -*- coding: utf-8 -*-
"""
debug code
==========

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import transfer_functions as tf
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2026 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def f_func_reference(
        x: float,
        x1: float,
        x2: float,
        y1: float,
        y2: float,
        m1: float,
        m2: float):
    """
    """
    m1_hat = (x2 - x1) * m1
    m2_hat = (x2 - x1) * m2
    c3 = 2*y1 + m1_hat - 2*y2 + m2_hat
    c2 = -3*y1 + 3*y2 - 2*m1_hat - m2_hat
    c1 = m1_hat
    c0 = y1
    t = (x - x1)/(x2 - x1)

    return c3 * (t**3) + c2 * (t**2) + c1 * t + c0


def _check_f_func_reference():
    x1 = 0
    x2 = 1.0
    y1 = 0.0
    y2 = 1.0
    m1 = 1.0
    m2 = 1.0
    x = np.linspace(x1, x2, 256)

    y = f_func_reference(x, x1, x2, y1, y2, m1, m2)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Title",
        graph_title_size=None,
        xlabel="X Axis Label",
        ylabel="Y Axis Label",
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
    pu.show_and_save(fig=fig, legend_loc='upper left', save_fname=None, show=True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    _check_f_func_reference()
