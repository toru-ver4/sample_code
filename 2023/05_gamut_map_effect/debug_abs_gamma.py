# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import plot_utility as pu


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def plot_gm24_wide_range():
    x = np.linspace(-1, 1, 256)
    sign_x = np.sign(x)
    y = (np.abs(x) ** (1/2.4)) * sign_x

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Gamma 2.4?",
        graph_title_size=None,
        xlabel="x",
        ylabel="y",
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
    ax1.plot(x, y, label="Gamma 2.4")
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname="./img/gm24.png")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_gm24_wide_range()
