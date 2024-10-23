# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.io import read_image

# import my libraries
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def test_plot(eotf_str="Gamma-2.4"):
    num_of_sample = 1024
    fname = f"./render_out/eotf_{eotf_str}_00086400.exr"
    img = read_image(fname)
    x = np.arange(num_of_sample)
    y = img[0, :num_of_sample, 1]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"EOTF",
        graph_title_size=None,
        xlabel="Input Code Value (10-bit)",
        ylabel="Linear Value",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x * 128 for x in range(8)] + [1023],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y, label=f"{eotf_str}")

    fname = f"./fugure/eotf.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_plot()
