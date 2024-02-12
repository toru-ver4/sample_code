# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries

# import my libraries
import plot_utility as pu
from ty_display_pro_hl import read_measure_result, calculate_elapsed_seconds


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def plot_peak_60s_data():
    csv_name = "./measure_result/measure_result_peak_60s.csv"
    read_data = read_measure_result(csv_name=csv_name)
    elapsed_time = calculate_elapsed_seconds(file_path=csv_name)
    luminance = read_data[..., 3]
    fname = "./img/peak_60s.png"

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel="Elapsed Time [sec]",
        ylabel="Luminance [nits]",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=[0, 1050],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(elapsed_time, luminance, '-o', label="3% Window")

    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='lower right', save_fname=fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_peak_60s_data()
