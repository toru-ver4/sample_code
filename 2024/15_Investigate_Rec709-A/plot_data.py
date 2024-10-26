# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.io import read_image
from colour.models import oetf_inverse_BT709, oetf_BT709

# import my libraries
import plot_utility as pu
import test_pattern_generator2 as tpg

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
    

def read_data_from_exr(eotf_str, num_of_sample):
    fname = f"./render_out/eotf_{eotf_str}_00086400.exr"
    img = read_image(fname)
    y = img[0, :num_of_sample, 1]

    return y


def read_data_from_tif(oetf_str, num_of_sample):
    fname = f"./render_out/oetf_{oetf_str}_00086400.tif"
    img = read_image(fname)
    y = img[0, :num_of_sample, 1]

    return y


def eotfs_plot():
    num_of_sample = 1024
    eotf_str_list = [
        "Rec.709",
        "Rec.709-A",
        "Gamma 2.2",
    ]
    color_list = [
        pu.BLUE,
        pu.GREEN,
        pu.RED,
    ]

    x2 = np.linspace(0, 1, num_of_sample)
    rec709_oetf_inv = oetf_inverse_BT709(x2)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(9, 9),
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
        ytick=[x * 0.1 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=4,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # pu.log_sacle_settings_x_linear_y_log(ax=ax1)

    for idx, eotf_str in enumerate(eotf_str_list):
        x = np.arange(num_of_sample)
        y = read_data_from_exr(eotf_str=eotf_str, num_of_sample=num_of_sample)
        ax1.plot(x, y, color=color_list[idx], label=f"Resolve {eotf_str}")
    ax1.plot(
        x, rec709_oetf_inv, '--', lw=1.5, color=pu.ORANGE,
        label="ITU-R BT.709 OETF Inverse")
    ax1.plot(
        x, x2 ** 1.961, '--k', lw=1.5, label="Gamma 1.961")

    fname = f"./fugure/eotfs_plot.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=False, save_fname=fname)


def eotfs_plot_low():
    num_of_sample = 1024
    eotf_str_list = [
        "Rec.709",
        "Rec.709-A",
        "Gamma 2.2",
    ]
    color_list = [
        pu.BLUE,
        pu.GREEN,
        pu.RED,
    ]
    x2 = np.linspace(0, 1, num_of_sample)
    rec709_oetf_inv = oetf_inverse_BT709(x2)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(9, 9),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"EOTF",
        graph_title_size=None,
        xlabel="Input Code Value (10-bit)",
        ylabel="Linear Value",
        axis_label_size=None,
        legend_size=17,
        xlim=[-20, 404],
        ylim=[-0.01, 0.18],
        xtick=[x * 64 for x in range(7)],
        ytick=[x * 0.025 for x in range(8)],
        xtick_size=None, ytick_size=None,
        linewidth=5,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # pu.log_sacle_settings_x_linear_y_log(ax=ax1)

    for idx, eotf_str in enumerate(eotf_str_list):
        x = np.arange(num_of_sample)
        y = read_data_from_exr(eotf_str=eotf_str, num_of_sample=num_of_sample)
        ax1.plot(x, y, color=color_list[idx], label=f"Resolve {eotf_str}")
    ax1.plot(
        x, rec709_oetf_inv, '--', lw=1.5, color=pu.ORANGE,
        label="ITU-R BT.709 OETF Inverse")
    ax1.plot(
        x, x2 ** 1.961, '--k', lw=1.5, label="Gamma 1.961")

    fname = f"./fugure/eotfs_plot_low.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=False, save_fname=fname
    )


def oetfs_plot():
    num_of_sample = 1280
    oetf_str_list = [
        "Rec.709",
        "Rec.709-A",
        "Gamma 2.2",
    ]
    color_list = [
        pu.BLUE,
        pu.GREEN,
        pu.RED,
    ]

    ref_val = 0.18
    max_exp = np.log2(1 / 0.18)
    min_exp = np.log2(0.00003 / 0.18)
    x = tpg.get_log2_x_scale(
        sample_num=num_of_sample, ref_val=ref_val,
        min_exposure=min_exp, max_exposure=max_exp)

    rec709_oetf_inv = oetf_BT709(x) * 1023

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(9, 9),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"OETF",
        graph_title_size=None,
        xlabel="Linear Value",
        ylabel="Code Value (10-bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x * 0.1 for x in range(11)],
        ytick=[x * 128 for x in range(8)] + [1023],
        xtick_size=None, ytick_size=None,
        linewidth=4,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # pu.log_sacle_settings_x_linear_y_log(ax=ax1)

    for idx, oetf_str in enumerate(oetf_str_list):
        # x = np.arange(num_of_sample)
        y = read_data_from_tif(oetf_str=oetf_str, num_of_sample=num_of_sample)
        y_10bit = y * 1023
        ax1.plot(
            x, y_10bit, color=color_list[idx], label=f"Resolve {oetf_str}")
    ax1.plot(
        x, rec709_oetf_inv, '--', lw=1.5, color=pu.ORANGE,
        label="ITU-R BT.709 OETF")
    ax1.plot(
        x, (x ** (1/1.961)) * 1023, '--k', lw=1.5, label="Gamma 1/1.961")

    fname = f"./fugure/oetfs_plot.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='lower right', show=False, save_fname=fname
    )


def oetfs_plot_low():
    num_of_sample = 1280
    oetf_str_list = [
        "Rec.709",
        "Rec.709-A",
        "Gamma 2.2",
    ]
    color_list = [
        pu.BLUE,
        pu.GREEN,
        pu.RED,
    ]

    ref_val = 0.18
    max_exp = np.log2(1 / 0.18)
    min_exp = np.log2(0.00003 / 0.18)
    x = tpg.get_log2_x_scale(
        sample_num=num_of_sample, ref_val=ref_val,
        min_exposure=min_exp, max_exposure=max_exp)

    rec709_oetf_inv = oetf_BT709(x) * 1023

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(9, 9),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"OETF",
        graph_title_size=None,
        xlabel="Linear Value",
        ylabel="Code Value (10-bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=[-0.01, 0.18],
        ylim=[-20, 404],
        xtick=[x * 0.025 for x in range(9)],
        ytick=[x * 64 for x in range(7)],
        xtick_size=None, ytick_size=None,
        linewidth=4,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # pu.log_sacle_settings_x_linear_y_log(ax=ax1)

    for idx, oetf_str in enumerate(oetf_str_list):
        # x = np.arange(num_of_sample)
        y = read_data_from_tif(oetf_str=oetf_str, num_of_sample=num_of_sample)
        y_10bit = y * 1023
        ax1.plot(
            x, y_10bit, color=color_list[idx], label=f"Resolve {oetf_str}")
    ax1.plot(
        x, rec709_oetf_inv, '--', lw=1.5, color=pu.ORANGE,
        label="ITU-R BT.709 OETF")
    ax1.plot(
        x, (x ** (1/1.961)) * 1023, '--k', lw=1.5, label="Gamma 1/1.961")

    fname = f"./fugure/oetfs_plot_low.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='lower right', show=False, save_fname=fname
    )


def check_rec709_a_gamma():
    num_of_sample = 1024
    x = np.linspace(0, 1, num_of_sample)
    y = read_data_from_exr(eotf_str="Rec.709-A", num_of_sample=num_of_sample)
    gamma = np.log(y[1:-1]) / np.log(x[1:-1])
    
    print(f"mean = {np.mean(gamma)}")
    print(f"var = {np.var(gamma)}")
    print(f"std * 3 = {np.std(gamma) * 3}")
    print(f"min = {np.min(gamma)}, max = {np.max(gamma)}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # test_plot()

    eotfs_plot()
    eotfs_plot_low()

    oetfs_plot()
    oetfs_plot_low()
    # check_rec709_a_gamma()
