# -*- coding: utf-8 -*-
"""
CIELAB の Gamut Boundary データの補間
=====================================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# import my libraries
import plot_utility as pu
import interpolate_cielab_data as icd

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


BT709_BOUNDARY = "./boundary_data/Chroma_BT709_l_256_h_256.npy"
BT2020_BOUNDAY = "./boundary_data/Chroma_BT2020_l_256_h_256.npy"


def plot_lc_plane(hue=0/360*2*np.pi):
    """
    とりあえず L*C* 平面をプロット
    """
    sample_num = 1024
    lstar = np.linspace(0, 100, sample_num)
    hue_list = np.ones_like(lstar) * hue
    lh = np.dstack([lstar, hue_list])
    lut_bt709 = np.load(BT709_BOUNDARY)
    lut_bt2020 = np.load(BT2020_BOUNDAY)
    chroma_bt709 = icd.bilinear_interpolation(lh, lut_bt709)
    chroma_bt2020 = icd.bilinear_interpolation(lh, lut_bt2020)

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=f"HUE = {hue/2/np.pi*360:.1f}°",
        graph_title_size=None,
        xlabel="Chroma",
        ylabel="Lightness",
        axis_label_size=None,
        legend_size=17,
        xlim=[-6, 200],
        ylim=[-3, 103],
        xtick=[x * 20 for x in range(11)],
        ytick=[x * 10 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(chroma_bt709, lstar, label="BT.709")
    ax1.plot(chroma_bt2020, lstar, label="BT.2020")
    graph_name = f"./figure/HUE = {hue/2/np.pi*360:.1f}.png"
    plt.legend(loc='lower right')
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


def check_lc_plane():
    for hue in np.linspace(0, 360, 12, endpoint=False):
        plot_lc_plane(hue/360*2*np.pi)


def calc_l_cusp():
    # calc_l_cusp_specific_hue()


def main_func():
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # check_lc_plane()
    main_func()
