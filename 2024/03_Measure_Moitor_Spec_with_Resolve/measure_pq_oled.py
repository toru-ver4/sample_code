# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path
import subprocess

# import third-party libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# import my libraries
import plot_utility as pu
from ty_display_pro_hl import read_xyz

matplotlib.use('TkAgg')

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def parse_ccss_file(file_path):
    def get_wavelength(file_path):
        with open(file_path, 'r') as file:
            data_lines = False
            wavelengths = []
            for line in file:
                line = line.strip()
                # BEGIN_DATA_FORMAT セクションの開始を検出
                if line == "BEGIN_DATA_FORMAT":
                    data_lines = True
                    continue
                # END_DATA_FORMAT セクションの終了を検出
                if line == "END_DATA_FORMAT":
                    data_lines = False
                    continue
                # データセクション内の行を処理
                if data_lines:
                    # SAMPLE_ID を除外し、SPEC_XXX を XXX に変換
                    parts = line.split()
                    if parts[0] == "SAMPLE_ID":
                        wavelengths = [int(p.split('_')[1]) for p in parts[1:]]
                        wavelengths = np.array(wavelengths, dtype=np.uint16)
                    break
        return wavelengths

    def get_spectrum_data(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data_start = lines.index("BEGIN_DATA\n") + 1
        data_end = lines.index("END_DATA\n")

        # 数値データとIDを格納するためのリスト
        data = []
        ids = []

        for line in lines[data_start:data_end]:
            parts = line.split()
            if len(parts) < 2:  # 少なくともIDと1つのデータが必要
                continue

            # IDとデータを分離
            ids.append(parts[0])
            data_values = np.array(parts[1:], dtype=float)
            data.append(data_values)

        return ids, np.array(data)

    wavelength = get_wavelength(file_path)
    spectrum_id, spectrum_data = get_spectrum_data(file_path)

    return wavelength, spectrum_id, spectrum_data


def analyze_ccss_file_all():
    ccss_file_list = [
        "./ccss/OLEDFamily_20Jul12.ccss",
        "./ccss/RG_Phosphor_Family_25Jul12.ccss",
        "./ccss/RGBLEDFamily_07Feb11.ccss",
        "./ccss/WGCCFLFamily_07Feb11.ccss",
        "./ccss/WLEDFamily_07Feb11.ccss"
    ]
    for ccss_file in ccss_file_list:
        wl, spd_id, spd = parse_ccss_file(file_path=ccss_file)
        ccss_name = Path(ccss_file).stem
        plot_spectrum_sample_each_with_qd_spd(
            wl=wl, spd_id=spd_id, spd=spd, ccss_name=ccss_name)


def plot_spectrum_sample_each_with_qd_spd(wl, spd_id, spd, ccss_name):
    y_max = np.max(spd) * 1.005
    y_min = -y_max * 0.02
    dell_spd = np.loadtxt(
        "./AW3225QF/WRGB_Data.csv", skiprows=1, usecols=(0, 1, 2, 3, 4),
        delimiter=',')
    x_dell = dell_spd[..., 0]
    y_dell = (dell_spd[..., 1] / np.max(dell_spd[..., 1])) * y_max
    fig, axes = plt.subplots(
        nrows=len(spd), ncols=1, figsize=(5, 24))

    for idx, y in enumerate(spd):
        ax = axes[idx]
        label_text = f"{ccss_name}-{spd_id[idx]}"
        ax.plot(wl, y, '-k', label=label_text)
        ax.plot(x_dell, y_dell, '--k', label="DELL G3223Q")
        # ax.set_title(ccss_name)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Power??")
        ax.legend(loc='upper right')

    plt.tight_layout()
    fname = f"./img/{ccss_name}_all_with_AW3225QF.png"
    print(fname)
    plt.savefig(fname)
    # plt.show()


def plot_aw3225qf_spectrum():
    dell_spd = np.loadtxt(
        "./AW3225QF/WRGB_Data.csv", skiprows=1, usecols=(0, 1, 2, 3, 4),
        delimiter=',')
    x = dell_spd[..., 0]
    w = (dell_spd[..., 1] / np.max(dell_spd))
    r = (dell_spd[..., 2] / np.max(dell_spd))
    g = (dell_spd[..., 3] / np.max(dell_spd))
    b = (dell_spd[..., 4] / np.max(dell_spd))
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="AW3225QF Spectral Power Distribution",
        graph_title_size=None,
        xlabel="Wavelength [nm]",
        ylabel="Relative Power",
        axis_label_size=None,
        legend_size=17,
        xlim=[380, 730],
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=5)
    ax1.plot(x, r, color=pu.RED, label="R")
    ax1.plot(x, g, color=pu.GREEN, label="G")
    ax1.plot(x, b, color=pu.BLUE, label="B")
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=True,
        save_fname="./img/AW3225QF_Spectrum.png")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # analyze_ccss_file_all()
    # plot_aw3225qf_spectrum()
