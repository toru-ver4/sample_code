# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import subprocess
import shutil
from pathlib import Path

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# import my libraries
from ty_utility import search_specific_extension_files
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def convert_from_edr_to_ccss(fname):
    """
    Converts a file from EDR format to CCSS format using 'oeminst.exe'.

    This function checks if 'oeminst.exe' exists in the system's PATH. If not
    found, FileNotFoundError is raised. Otherwise, it executes the conversion
    using 'oeminst.exe' with appropriate command-line options.

    Parameters
    ----------
    fname : str
        The name (or path) of the EDR file to be converted.

    Raises
    ------
    FileNotFoundError
        If 'oeminst.exe' is not found in the system's PATH.

    Examples
    --------
    >>> convert_from_edr_to_ccss("example.edr")
    # This will attempt to convert 'example.edr' using 'oeminst.exe'.

    Notes
    -----
    Ensure 'oeminst.exe' is installed and available in your system's PATH.
    """
    if shutil.which("oeminst.exe") is None:
        raise FileNotFoundError("oeminst.exe not found in system PATH.")

    cmd_plus_ops = ["oeminst.exe", '-c', fname]
    subprocess.run(cmd_plus_ops)


def create_ccss_files():
    cwd_bak = os.getcwd()
    os.chdir("./ccss")
    edr_file_list = search_specific_extension_files(
        dir="../edr", ext=".edr")
    for edr_file in edr_file_list:
        convert_from_edr_to_ccss(edr_file)
    os.chdir(cwd_bak)


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


def parse_ccss_data(file_content):
    lines = file_content.split("\n")
    data_start = lines.index("BEGIN_DATA") + 1
    data_end = lines.index("END_DATA")

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

    return np.array(ids), np.array(data)


def plot_spectrum_sample_at_the_same_time(wl, spd_id, spd, ccss_name):
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(14, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"{ccss_name}",
        graph_title_size=None,
        xlabel="Wavelength [nm]",
        ylabel="Power??",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=2,
        minor_xtick_num=None,
        minor_ytick_num=None)
    for idx, y in enumerate(spd):
        ax1.plot(wl, y, label=spd_id[idx])
    pu.show_and_save(
        fig=fig, legend_loc='upper right', save_fname=None, show=True)


def plot_spectrum_sample_each(wl, spd_id, spd, ccss_name):
    y_max = np.max(spd) * 1.005
    y_min = -y_max * 0.02
    fig, axes = plt.subplots(
        nrows=len(spd), ncols=1, figsize=(6, 24))

    for idx, y in enumerate(spd):
        ax = axes[idx]
        ax.plot(wl, y, label=spd_id[idx])
        # ax.set_title(ccss_name)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Power??")
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("./hoge.png")
    plt.show()


def analyze_ccss_file_all():
    ccss_file_list = search_specific_extension_files(
        dir="./ccss", ext=".ccss")
    for ccss_file in ccss_file_list:
        wl, spd_id, spd = parse_ccss_file(file_path=ccss_file)
        # print(wl)
        ccss_name = Path(ccss_file).stem
        plot_spectrum_sample_each(
            wl=wl, spd_id=spd_id, spd=spd, ccss_name=ccss_name)
        print(spd.shape)
        break


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_ccss_files()
    # analyze_ccss_file_all()

    ccss_file = "./ccss/CRT.ccss"
    wl, spd_id, spd = parse_ccss_file(file_path=ccss_file)
    # print(wl)
    ccss_name = Path(ccss_file).stem
    plot_spectrum_sample_each(
        wl=wl, spd_id=spd_id, spd=spd, ccss_name=ccss_name)
    print(spd.shape)
