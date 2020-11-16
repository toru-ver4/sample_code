# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from colour import write_image, read_image

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_ramp():
    x = np.linspace(0, 1, 1920).reshape((1, 1920, 1))

    img = np.ones((1080, 1920, 3))
    img = x * img
    write_image(img, "test_src.tif", bit_depth='uint16')


def check_input_drt_test():
    create_ramp()
    file_list = [
        ['./img/test_out_sdr100.tif', 'SDR100'],
        ['./img/test_out_hdr500.tif', 'HDR500'],
        ['./img/test_out_hdr1000.tif', 'HDR1000'],
        ['./img/test_out_hdr2000.tif', 'HDR2000'],
        ['./img/test_out_hdr4000.tif', 'HDR4000'],
        ['./img/test_out_off.tif', 'DRT_OFF']
    ]
    x = np.linspace(0, 1, 1920)
    x_luminance = tf.eotf_to_luminance(x, tf.ST2084)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="DaVinci17 Input DRT Characteristics",
        graph_title_size=None,
        xlabel="Input Luminance [cd/m2]",
        ylabel="Output Luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=[0.0009, 15000],
        ylim=[0.0009, 15000],
        xtick=None,
        ytick=None,
        xtick_size=None,
        ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    pu.log_scale_settings(ax1, grid_alpha=0.5, bg_color="#E0E0E0")

    for idx in range(len(file_list)):
        img = read_image(file_list[idx][0])[0, :, 0]
        label = file_list[idx][1]
        y_luminance = tf.eotf_to_luminance(img, tf.ST2084)
        ax1.plot(x_luminance, y_luminance, label=label)

    plt.legend(loc='upper left')
    plt.savefig("input_drt_spec.png", bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def main_func():
    check_input_drt_test()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
