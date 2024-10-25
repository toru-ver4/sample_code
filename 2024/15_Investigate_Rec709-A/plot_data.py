# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.io import read_image
from colour.models import oetf_inverse_BT709

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
    

def read_data_from_exr(eotf_str, num_of_sample):
    fname = f"./render_out/eotf_{eotf_str}_00086400.exr"
    img = read_image(fname)
    y = img[0, :num_of_sample, 1]

    return y

def eotfs_plot():
    num_of_sample = 1024
    eotf_str_list = [
        "Gamma 2.2",
        # "Gamma 2.4",
        "Rec.709-A",
        "Rec.709",
    ]
    color_list = [
        pu.RED,
        pu.GREEN,
        pu.BLUE,
        pu.ORANGE
    ]
    x2 = np.linspace(0, 1, num_of_sample)
    rec709_oetf_inv = oetf_inverse_BT709(x2)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"EOTF",
        graph_title_size=None,
        xlabel="Input Code Value (10-bit)",
        ylabel="Linear Value",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=[0.00001, 1.1],
        xtick=[x * 128 for x in range(8)] + [1023],
        # ytick=[x * 0.1 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=5,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_sacle_settings_x_linear_y_log(ax=ax1)
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
        fig=fig, legend_loc='lower right', show=False,
        save_fname=fname)
    
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
    # eotfs_plot()
    check_rec709_a_gamma()
