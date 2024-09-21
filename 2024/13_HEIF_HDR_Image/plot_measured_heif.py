# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path
import re

# import third-party libraries
import numpy as np

# import my libraries
from ty_display_pro_hl import read_measure_result
from create_data_for_heif_hdr import create_luminance_value_array
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def plot_each_app_on_specific_device(csv_list, num_of_measure=33):
    ref_luminance = create_luminance_value_array(num_of_array=num_of_measure)
    measured_list = [read_measure_result(csv_name) for csv_name in csv_list]

    xy_min = 0.09
    xy_max = 10100
    title = re.match(r"(\w*?)_", Path(csv_list[0]).stem).group(1)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"{title} Characteristics",
        graph_title_size=None,
        xlabel="Target Luminance (nits)",
        ylabel="Measured Luminance (nits)",
        axis_label_size=None,
        legend_size=17,
        xlim=(xy_min, xy_max),
        ylim=(xy_min, xy_max),
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_scale_settings(ax1=ax1)

    ax1.plot(ref_luminance, ref_luminance, 'k--', label="Reference")
    for measured, csv_name in zip(measured_list, csv_list):
        xx = ref_luminance
        yy = measured[..., 1]
        match = re.search(r".+_([^_.]+)", Path(csv_name).stem)
        label = match.group(1)
        ax1.plot(xx, yy, "-o", label=label)

    graph_fname = f"./img/{title}.png"
    print(graph_fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=True,
        save_fname=graph_fname)


def plot_comparison_iOS_and_macOS_for_photos_app():
    ref_luminance = create_luminance_value_array(num_of_array=33)
    ios_result = read_measure_result("./mesured_luminance/iPhone_Photos.csv")
    macos_result = read_measure_result("./mesured_luminance/MBP_Photos.csv")

    xy_min = 0.9
    xy_max = 10100

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Display Characteristics of HEIC Files",
        graph_title_size=None,
        xlabel="Target Luminance (nits)",
        ylabel="Measured Luminance (nits)",
        axis_label_size=None,
        legend_size=17,
        xlim=(xy_min, xy_max),
        ylim=(xy_min, xy_max),
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_scale_settings(ax1=ax1)

    ax1.plot(ref_luminance, ref_luminance, 'k--', label="Reference")
    ax1.plot(
        ref_luminance, ios_result[..., 1], "-o",
        label="iPhone 15 Pro Max, Photos app"
    )
    ax1.plot(
        ref_luminance, macos_result[..., 1], "-o",
        label="MacBook Pro (16-inch, M1, XDR), Photos app"
    )

    graph_fname = f"./img/figure1.png"
    print(graph_fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=True, save_fname=graph_fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # iphone_csv_list = [
    #     "./mesured_luminance/iPhone_Files.csv",
    #     "./mesured_luminance/iPhone_YouTube.csv",
    #     "./mesured_luminance/iPhone_Photos.csv",
    # ]
    # plot_each_app_on_specific_device(
    #     csv_list=iphone_csv_list, num_of_measure=33
    # )

    # mbp_csv_list = [
    #     "./mesured_luminance/MBP_QuickTime.csv",
    #     "./mesured_luminance/MBP_YouTube.csv",
    #     "./mesured_luminance/MBP_Photos.csv",
    # ]
    # plot_each_app_on_specific_device(
    #     csv_list=mbp_csv_list, num_of_measure=33
    # )
    plot_comparison_iOS_and_macOS_for_photos_app()
