# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import subprocess
import shutil
from pathlib import Path
import re

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from colour import XYZ_to_Lab, xyY_to_XYZ, XYZ_to_xyY
from colour.difference import delta_E_CIE2000

# import my libraries
from ty_utility import search_specific_extension_files
import plot_utility as pu
import test_pattern_generator2 as tpg
from display_pro_hl import read_measure_result, read_xyz_and_save_to_csv_file
import transfer_functions as tf
import color_space as cs

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

    print(f"oeminst.exe -c {fname}")
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
        nrows=len(spd), ncols=1, figsize=(5, 24))

    for idx, y in enumerate(spd):
        ax = axes[idx]
        label_text = f"{ccss_name}-No.{spd_id[idx]}"
        ax.plot(wl, y, '-k', label=label_text)
        # ax.set_title(ccss_name)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Power??")
        ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    fname = f"./img/{ccss_name}_all.png"
    print(fname)
    plt.savefig(fname)
    # plt.show()


def plot_spectrum_sample_each_with_dell_spd(wl, spd_id, spd, ccss_name):
    y_max = np.max(spd) * 1.005
    y_min = -y_max * 0.02
    dell_spd = np.loadtxt(
        "./DELL_LCD_Spectrum.csv", skiprows=5, usecols=(0, 1), delimiter=',')
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
    fname = f"./img/{ccss_name}_all_with_G3223Q.png"
    print(fname)
    plt.savefig(fname)
    # plt.show()


def plot_spectrum_sample_each_with_peak(wl, spd_id, spd, ccss_name):
    y_max = np.max(spd) * 1.005
    y_min = -y_max * 0.02

    peak_idx = [np.argmax(spd[ii + 1]) for ii in range(3)]
    peak_xy = [
        np.array([[wl[peak_idx[ii]], y_min], [wl[peak_idx[ii]], y_max]])
        for ii in range(3)]
    color_list = [pu.RED, pu.GREEN, pu.BLUE]
    print(peak_xy)

    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(8, 10))

    for idx in range(3):
        y = spd[idx * 4]
        ax = axes[idx]
        label_text = f"{ccss_name}-{spd_id[idx]}"
        ax.plot(wl, y, '-k', label=label_text)
        for ii in range(3):
            ax.plot(
                peak_xy[ii][..., 0], peak_xy[ii][..., 1], '-',
                color=color_list[ii], label=None)
        # ax.set_title(ccss_name)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Power??")
        ax.legend(loc='upper right')

    plt.tight_layout()
    fname = f"./img/{ccss_name}_all_with_peak.png"
    print(fname)
    plt.savefig(fname)
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
        # plot_spectrum_sample_each_with_dell_spd(
        #     wl=wl, spd_id=spd_id, spd=spd, ccss_name=ccss_name)


def concat_all_img():
    file_list = search_specific_extension_files(dir="./img", ext=".png")
    img_buf = []

    for file_name in file_list:
        img_temp = tpg.img_read_as_float(str(file_name))
        img_buf.append(img_temp)

    img_out = np.hstack(img_buf)

    tpg.img_wirte_float_as_16bit_int("./concat_img.png", img_out)


def concat_all_img_with_G3223Q():
    file_list = search_specific_extension_files(dir="./img", ext="G3223Q.png")
    img_buf = []

    for file_name in file_list:
        img_temp = tpg.img_read_as_float(str(file_name.resolve()))
        img_buf.append(img_temp)

    img_out = np.hstack(img_buf)

    tpg.img_wirte_float_as_16bit_int("./concat_img_with_G3223Q.png", img_out)


def calc_color_checker_xyz_value_from_tp(fname):
    st_pos_h = 350
    ed_pos_h = 1600
    st_pos_v = 170
    ed_pos_v = 926
    h_len = 6
    v_len = 4
    h_step = int((ed_pos_h - st_pos_h) / (h_len - 1))
    v_step = int((ed_pos_v - st_pos_v) / (v_len - 1))

    pos_info = []
    for v_idx in range(v_len):
        pos_v = st_pos_v + v_idx * v_step
        for h_idx in range(h_len):
            pos_h = st_pos_h + h_idx * h_step
            pos_info.append([pos_h, pos_v])

    img = tpg.img_read_as_float(fname)
    img = tf.eotf(img, tf.GAMMA24)
    cc_rgb = np.array([img[pos[1], pos[0]] for pos in pos_info])
    cc_xyz = cs.rgb_to_large_xyz(
        rgb=cc_rgb, color_space_name=cs.BT709, xyz_white=cs.D65)

    return cc_xyz


def calc_delta_xyY(
        ref_xyz: np.ndarray, measured_xyz: np.ndarray,
        csv_name: str):
    ref_xyY = XYZ_to_xyY(ref_xyz)
    measured_xyY = XYZ_to_xyY(measured_xyz)
    diff_xyY = ref_xyY - measured_xyY

    with open(csv_name, 'wt') as f:
        buf = ""
        buf += "No,ref_x,ref_y,ref_Y,x,y,Y,diff_x,diff_y,diff_Y\n"
        for ii in range(len(ref_xyY)):
            rr_xyY = ref_xyY[ii]
            mm_xyY = measured_xyY[ii]
            dd_xyY = diff_xyY[ii]

            buf += f"{ii},"
            buf += f"{rr_xyY[0]},{rr_xyY[1]},{rr_xyY[2]},"
            buf += f"{mm_xyY[0]},{mm_xyY[1]},{mm_xyY[2]},"
            buf += f"{dd_xyY[0]},{dd_xyY[1]},{dd_xyY[2]}\n"

        f.write(buf)


def calc_colorchecker_de2000(measured_xyz, ref_luminance):
    measured_lab = XYZ_to_Lab(measured_xyz / ref_luminance)

    target_xyz = calc_color_checker_xyz_value_from_tp(
        fname="../../2020/026_create_icc_profiles/img/bt709_img_with_icc.png")
    # target_xyz = xyY_to_XYZ(tpg.generate_color_checker_xyY_value())
    # target_xyz *= ref_luminance
    # print(target_xyz)
    target_lab = XYZ_to_Lab(target_xyz)

    de2000 = delta_E_CIE2000(target_lab, measured_lab)

    return de2000


def plot_colorchecker_de2000(
        de2k: np.ndarray, max_diff: float, title_suffix=""):
    x = np.arange(len(de2k)) + 1
    bar_color = np.clip(tpg.generate_color_checker_rgb_value(), 0, 1)
    bar_color = tf.oetf(bar_color, tf.SRGB)
    title = "Color Difference" + " " + title_suffix
    fname = f"./img/{title}.png"
    average = np.mean(de2k)
    min_val = np.min(de2k)
    max_val = np.max(de2k)
    info_text = f'Average: {average:.2f}, '
    info_text += f'Min: {min_val:.2f}, Max: {max_val:.2f}'
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(14, 6),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=title,
        graph_title_size=None,
        xlabel="ColorChecker Index",
        ylabel="ΔE2000",
        axis_label_size=None,
        legend_size=17,
        xlim=[0, 25],
        ylim=[0, max_diff * 1.05],
        xtick=np.arange(24) + 1,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.bar(x, de2k, color=bar_color, edgecolor='k', width=0.6, label=None)
    ax1.grid(False, axis='x')

    # add info text
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    text_pos_x = xmin + (xmax - xmin) * 0.02
    text_pos_y = ymax - (ymax - ymin) * 0.05
    bbox_ops = dict(
        facecolor='white', edgecolor='black', boxstyle='square,pad=0.5')
    ax1.text(
        text_pos_x, text_pos_y, info_text, fontsize=20, va='top', ha='left',
        bbox=bbox_ops)

    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, save_fname=fname, show=False)


def plot_colorchecker_de2000_with_PROFILER(
        de2k: np.ndarray, title_suffix=""):
    x = np.arange(len(de2k)) + 1
    y_ref = np.array(
        [0.71, 0.49, 0.15, 0.49, 0.26, 0.25,
         0.28, 0.21, 0.32, 0.32, 0.07, 0.13,
         0.05, 0.23, 0.12, 0.31, 0.34, 0.23,
         0.78, 0.5, 1.18, 0.2, 0.94, 0.36])
    bar_color = np.clip(tpg.generate_color_checker_rgb_value(), 0, 1)
    bar_color = tf.oetf(bar_color, tf.SRGB)
    title = "Color Difference" + " " + title_suffix
    fname = f"./img/{title}_with_PROFILER.png"
    average = np.mean(de2k)
    min_val = np.min(de2k)
    max_val = np.max(de2k)
    info_text = f'Average: {average:.2f}, min: {min_val:.2f}, max: {max_val:.2f}'
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(14, 6),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=title,
        graph_title_size=None,
        xlabel="Index",
        ylabel="ΔE2000",
        axis_label_size=None,
        legend_size=17,
        xlim=[0, 25],
        ylim=None,
        xtick=np.arange(24) + 1,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.bar(x-0.15, de2k, color=bar_color, edgecolor='k', width=0.3, label=None)
    ax1.bar(x+0.15, y_ref, color=pu.GRAY90, edgecolor='k', width=0.3, label=None)
    ax1.grid(False, axis='x')

    # add info text
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    text_pos_x = xmin + (xmax - xmin) * 0.02
    text_pos_y = ymax - (ymax - ymin) * 0.05
    bbox_ops = dict(
        facecolor='white', edgecolor='black', boxstyle='square,pad=0.5')
    ax1.text(
        text_pos_x, text_pos_y, info_text, fontsize=20, va='top', ha='left',
        bbox=bbox_ops)

    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, save_fname=fname, show=True)


def check_diff_ccss_with_colorchecker(
        result_with_ccss, result_without_ccss, ccss_suffix=""):
    data_ccss = read_measure_result(csv_name=result_with_ccss)
    data_no_ccss = read_measure_result(csv_name=result_without_ccss)
    ref_luminance = 61

    data_ccss_xyz = data_ccss[..., :3]
    data_no_ccss_xyz = data_no_ccss[..., :3]

    de2000_ccss = calc_colorchecker_de2000(
        measured_xyz=data_ccss_xyz, ref_luminance=ref_luminance)
    de2000_no_ccss = calc_colorchecker_de2000(
        measured_xyz=data_no_ccss_xyz, ref_luminance=ref_luminance)
    max_diff = np.max(de2000_no_ccss)

    plot_colorchecker_de2000(
        de2k=de2000_ccss, max_diff=max_diff,
        title_suffix=f"with {ccss_suffix}")
    plot_colorchecker_de2000(
        de2k=de2000_no_ccss, max_diff=max_diff,
        title_suffix=f"without {ccss_suffix}")

    de2k_report = np.array([
        0.53, 0.2, 0.57, 0.25, 0.19, 0.27,
        0.22, 0.29, 0.12, 0.33, 0.57, 0.17,
        0.37, 0.09, 0.23, 0.21, 0.26, 0.48,
        0.75, 0.93, 0.7, 0.71, 1.1, 1.29
    ])
    plot_colorchecker_de2000(
        de2k=de2k_report, max_diff=max_diff,
        title_suffix="PROFILER Report")
    # plot_colorchecker_de2000_with_PROFILER(
    #     de2k=de2000_ccss, title_suffix=f"with {ccss_suffix}")

    # ref_xyz = calc_color_checker_xyz_value_from_tp(
    #     fname="../../2020/026_create_icc_profiles/img/bt709_img_with_icc.png")
    # calc_delta_xyY(
    #     ref_xyz=ref_xyz,
    #     measured_xyz=data_no_ccss_xyz/ref_luminance,
    #     csv_name=f"./{ccss_suffix}.csv")


def debug_check_de2000():
    xyY_no_ccss = np.array([0.63424544, 0.309768, 0.333399])
    xyY_ccss = np.array([0.62852337, 0.312340, 0.329347])
    xyY_ref = np.array([0.63, 0.3127, 0.3290])

    xyY_list = [xyY_ref, xyY_no_ccss, xyY_ccss]
    xyz_list = [xyY_to_XYZ(xyY) for xyY in xyY_list]
    lab_list = [XYZ_to_Lab(xyz) for xyz in xyz_list]

    lab_ref = lab_list[0]
    lab_no_ccss = lab_list[1]
    lab_ccss = lab_list[2]

    print(delta_E_CIE2000(lab_ref, lab_no_ccss))
    print(delta_E_CIE2000(lab_ref, lab_ccss))


def make_blog_ccss_file_name(idx, ccss_name):
    return f"./blog_img/{idx}_{ccss_name}_all.png"


def plot_blog_ccss_file(file_idx, wl, spd_id, spd, ccss_name):
    color_list = [pu.GRAY10, pu.RED, pu.GREEN, pu.BLUE]
    ccss_name_base = ccss_name.split("_")[0]
    y_max = np.max(spd) * 1.005
    y_min = -y_max * 0.02
    fig, axes = plt.subplots(
        nrows=len(spd), ncols=1, figsize=(4, 20))

    for idx, y in enumerate(spd):
        ax = axes[idx]
        label_text = f"{ccss_name_base}-No.{spd_id[idx]}"
        ax.plot(wl, y, color=color_list[idx % 4], label=label_text)
        # ax.set_title(ccss_name)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Power?")
        ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    fname = make_blog_ccss_file_name(file_idx, ccss_name)
    print(fname)
    plt.savefig(fname)


def plot_blog_ccss_files_all():
    ccss_file_list = [
        "./ccss/RGBLEDFamily_07Feb11.ccss", "./ccss/WGCCFLFamily_07Feb11.ccss",
        "./ccss/CCFLFamily_07Feb11.ccss", "./ccss/WLEDFamily_07Feb11.ccss"
    ]

    img_buf = []
    for idx, ccss_file in enumerate(ccss_file_list):
        wl, spd_id, spd = parse_ccss_file(file_path=ccss_file)
        # print(wl)
        ccss_name = Path(ccss_file).stem
        plot_blog_ccss_file(
            file_idx=idx, wl=wl, spd_id=spd_id, spd=spd, ccss_name=ccss_name)
        graph_img_name = make_blog_ccss_file_name(idx, ccss_name)
        print(graph_img_name)
        graph_img = tpg.img_read_as_float(graph_img_name)
        img_buf.append(graph_img)

    out_img = np.hstack(img_buf)
    out_fname = "./blog_img/concat_ccss_graph.png"
    print(out_fname)
    tpg.img_wirte_float_as_16bit_int(out_fname, out_img)


def check_ccss_difference_with_white_patch():
    measure_file_name = "./measure_result/ccss_difference.csv"
    ccss_file_list = search_specific_extension_files(
        dir="./ccss", ext=".ccss")
    marker_size = 12
    pattern = r"ccss\\|_[0-9]+[a-zA-Z]+[0-9]+\.ccss"
    print(ccss_file_list)

    # # remove .csv file before measurement start
    # file_path = Path(measure_file_name)
    # file_path.unlink(missing_ok=True)

    # for ccss_file in ccss_file_list:
    #     read_xyz_and_save_to_csv_file(
    #         result_fname=measure_file_name, ccss_file=ccss_file)

    data_list = read_measure_result(csv_name=measure_file_name)

    x_min = np.min(data_list[..., 4])
    x_max = np.max(data_list[..., 4])
    y_min = np.min(data_list[..., 5])
    y_max = np.max(data_list[..., 5])

    rad = np.linspace(0, 1, 1024) * 2 * np.pi
    x_01 = np.cos(rad) * 0.001 + cs.D65[0]
    x_02 = np.cos(rad) * 0.002 + cs.D65[0]
    y_01 = np.sin(rad) * 0.001 + cs.D65[1]
    y_02 = np.sin(rad) * 0.002 + cs.D65[1]

    x_diff = max(abs(cs.D65[0]-x_min), abs(cs.D65[0]-x_max)) * 1.2
    y_diff = max(abs(cs.D65[1]-y_min), abs(cs.D65[1]-y_max)) * 1.2
    diff_d65 = max(x_diff, y_diff)

    print(x_diff, y_diff)

    title = "The Relationship Between the CCSS File "
    title += "and the D65 Measurement Values"
    fig, ax1 = pu.plot_1_graph(
        fontsize=12,
        figsize=(9, 9),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=title,
        graph_title_size=None,
        xlabel="x",
        ylabel="y",
        axis_label_size=None,
        legend_size=17,
        xlim=[cs.D65[0]-diff_d65, cs.D65[0]+diff_d65],
        ylim=[cs.D65[1]-diff_d65, cs.D65[1]+diff_d65],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    for data, ccss_file in zip(data_list, ccss_file_list):
        x = data[..., 4]
        y = data[..., 5]
        label = re.sub(pattern, '', ccss_file)
        ax1.plot(x, y, 'o', ms=marker_size, label=label)
    ax1.plot(
        cs.D65[0], cs.D65[1], 'x', color='k', label="D65",
        ms=marker_size, mew=4)
    ax1.plot(x_01, y_01, '--k', label="Δ0.001", lw=2, alpha=0.66)
    ax1.plot(x_02, y_02, '-.k', label="Δ0.002", lw=2, alpha=0.66)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', fontsize=12,
        save_fname="./blog_img/d65_with_all_ccss.png", show=True)


def plot_wled_ccss_3graph():
    ccss_file = "./ccss/WLEDFamily_07Feb11.ccss"
    wl, spd_id, spd = parse_ccss_file(file_path=ccss_file)
    color_list = [pu.GRAY10, pu.RED, pu.GREEN, pu.BLUE]
    ccss_name_base = "WLEDFamily"
    y_max = np.max(spd) * 1.02
    y_min = -y_max * 0.02

    fig, axes = plt.subplots(
        nrows=4, ncols=3, figsize=(16, 10))

    for h_idx in range(3):
        for v_idx in range(4):
            idx = 4 * h_idx + v_idx
            y = spd[idx]
            ax = axes[v_idx, h_idx]
            label_text = f"{ccss_name_base}-No.{spd_id[idx]}"
            ax.plot(wl, y, color=color_list[idx % 4], label=label_text)
            # ax.set_title(ccss_name)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Wavelength [nm]")
            ax.set_ylabel("Power?")
            ax.legend(loc='upper right', fontsize=12)
            ax.grid(True, linestyle='--', color='gray', linewidth=0.5)

    plt.tight_layout()
    fname = "./blog_img/WLED_CCSS_BLOG.png"
    print(fname)
    plt.savefig(fname)
    plt.show()


def output_colorchecker_ref_value():
    target_xyz = calc_color_checker_xyz_value_from_tp(
        fname="../../2020/026_create_icc_profiles/img/bt709_img_with_icc.png")
    with open("./measure_result/colorchecker_ref_value.csv", 'wt') as f:
        buf = ""
        buf += "idx,X,Y,Z\n"
        for idx, xyz in enumerate(target_xyz):
            buf += f"{idx+1},{xyz[0]:.7e},{xyz[1]:.7e},{xyz[2]:.7e}\n"

        f.write(buf)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_ccss_files()
    # analyze_ccss_file_all()
    # concat_all_img()
    # concat_all_img_with_G3223Q()
    # check_diff_ccss_with_colorchecker(
    #     result_with_ccss="./measure_result/Calibration-WLED_with_ccss.py",
    #     result_without_ccss="./measure_result/Calibration-WLED_no_ccss.py",
    #     ccss_suffix="WLEDFamily.ccss"
    # )

    # plot_blog_ccss_files_all()

    # check_ccss_difference_with_white_patch()

    # plot_wled_ccss_3graph()

    output_colorchecker_ref_value()
