# -*- coding: utf-8 -*-
"""
plot BT.2446 Methoc C
======================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# import my libraries
import plot_utility as pu
import bt2446_method_c as bmc
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_hdr_ip_sub_line(x_min=0.1, y_min=0.1, y_hdr_ip=1, k1=1):
    x = [y_hdr_ip, y_hdr_ip, x_min]
    y = [y_min, k1 * y_hdr_ip, k1 * y_hdr_ip]

    return x, y


def calc_hdr_ref_sub_line(x_min=0.1, y_min=0.1, y_hdr_ref=1, y_sdr_wp=1):
    x = [y_hdr_ref, y_hdr_ref, x_min]
    y = [y_min, y_sdr_wp, y_sdr_wp]

    return x, y


def plot_tome_curve(k1=0.8, k3=0.7, y_sdr_ip=60, y_hdr_ref=203):
    x_min = 0.1
    y_min = 0.1
    y_hdr_ip, y_sdr_wp, k2, k4 = bmc.calc_tonemapping_parameters(
        k1=k1, k3=k3, y_sdr_ip=y_sdr_ip, y_hdr_ref=y_hdr_ref)
    x = tf.eotf_to_luminance(np.linspace(0, 1, 1024), tf.ST2084)
    # x = np.linspace(0, 10000, 1024)
    y = bmc.bt2446_method_c_tonemapping_core(
        x, k1=k1, k3=k3, y_sdr_ip=y_sdr_ip, y_hdr_ref=y_hdr_ref)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 6),
        graph_title="BT.2446 Method C",
        xlabel="HDR Luminance [cd/m2]",
        ylabel="SDR Luminance [cd/m2]",
        xlim=(x_min, 10000),
        ylim=(y_min, 250),
        linewidth=3,
        return_figure=True)
    pu.log_scale_settings(ax1)
    tonecurve = ax1.plot(x, y, label="BT2446 Method C")

    # annotation
    # bmc.draw_ip_wp_annotation(
    #     x_min, y_min, ax1, k1, y_hdr_ip, y_hdr_ref, y_sdr_wp, fontsize=16)

    # auxiliary line
    hdr_ip_line = ax1.plot(
        [y_hdr_ip, y_hdr_ip, x_min], [y_min, k1 * y_hdr_ip, k1 * y_hdr_ip],
        'k--', lw=2, c='#555555')
    hdr_ref_line = ax1.plot(
        [y_hdr_ref, y_hdr_ref, x_min], [y_min, y_sdr_wp, y_sdr_wp],
        'k--', lw=2, c='#555555')

    # plot youtube tonemap
    youtube = np.load("./luts/youtube.npy")
    ax1.plot(youtube[..., 0], youtube[..., 1], label="YouTube")
    plt.legend(loc='lower right', fontsize=14)

    return fig, ax1, [tonecurve[0], hdr_ip_line[0], hdr_ref_line[0]]


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_tome_curve(k1=0.8, k3=0.7, y_sdr_ip=60, y_hdr_ref=203)
    plt.show()
