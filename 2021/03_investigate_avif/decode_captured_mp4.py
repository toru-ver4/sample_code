# -*- coding: utf-8 -*-
"""
decode
======
"""

# import standard libraries
import os
import subprocess
from pathlib import Path

# import third-party libraries
import matplotlib.pyplot as plt
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg
import plot_utility as pu


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def decode_mp4(in_fname="./captured_video/capture_sample.mp4"):
    """
    decode video
    return single 10bit data
    """
    stem_name = Path(Path(in_fname).name).stem
    out_fname = f'./decoded_png/{stem_name}.png'
    cmd = "ffmpeg"
    ops = [
        '-i', in_fname, '-vframes', '1',
        str(out_fname), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)

    return out_fname


def extract_gray_patch_from_tp(img_name):
    gray_st_pos_h = 64
    gray_pos_v = 826
    gray_ed_pos_h = 1858
    gray_patch_num = 65
    max_level = 1023
    step = (max_level + 1) // (gray_patch_num - 1)
    level_list = [x * step for x in range(gray_patch_num - 1)] + [max_level]
    pos_h_list = np.uint16(
        np.round(np.linspace(gray_st_pos_h, gray_ed_pos_h, gray_patch_num)))
    pos_v_list = np.ones_like(pos_h_list) * gray_pos_v
    pos_list = np.dstack([pos_h_list, pos_v_list]).reshape((gray_patch_num, 2))

    img_float = tpg.img_read_as_float(img_name)
    img_10bit = np.uint16(np.round(img_float * 1023))

    buf = ""
    for idx, pos in enumerate(pos_list):
        rgb = img_10bit[pos[1], pos[0]]
        line_str = f"{level_list[idx]},{rgb[0]},{rgb[1]},{rgb[2]}\n"
        print(line_str.rstrip())
        buf += line_str

    stem_name = Path(Path(img_name).name).stem
    csv_name = f"./csv/{stem_name}.csv"
    with open(csv_name, 'wt') as f:
        f.write(buf)

    return csv_name


def plot_cautured_data_main(csv_file, graph_title):
    data = np.loadtxt(csv_file, delimiter=',')

    x = data[..., 0]
    rr = data[..., 1]
    gg = data[..., 2]
    bb = data[..., 3]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title=graph_title,
        graph_title_size=None,
        xlabel="Source Code Value (10bit)",
        ylabel="Captured Code Value (10bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=[-30, 1060],
        ylim=[-30, 1060],
        xtick=[x * 128 for x in range(8)] + [1023],
        ytick=[x * 128 for x in range(8)] + [1023],
        xtick_size=None, ytick_size=None,
        linewidth=2,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    ax1.plot(x, x, '-o', color='k', label="Expected value")
    ax1.plot(x, rr, '-o', color=pu.RED, label="R data")
    ax1.plot(x, gg, '-o', color=pu.GREEN, label="G data")
    ax1.plot(x, bb, '-o', color=pu.SKY, label="B data")
    plt.legend(loc='upper left')

    stem_name = Path(Path(csv_file).name).stem
    graph_name = f"./graph/{stem_name}.png"
    plt.savefig(graph_name, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def plot_captured_data(
        in_fname="./captured_video/capture_sample.mp4",
        graph_title="Code value of the captured AVIF"):
    """
    * decode video
    * capture 0, 16, 32, ..., 1023 CV
    * save captured data
    * plot captured data
    """
    decoded_img_name = decode_mp4(in_fname=in_fname)
    csv_name = extract_gray_patch_from_tp(img_name=decoded_img_name)
    plot_cautured_data_main(csv_name, graph_title)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_captured_data(in_fname="./captured_video/capture_sample.mp4")
    # plot_captured_data(in_fname="./captured_video/decklink_output.mp4")
    # plot_captured_data(in_fname="./captured_video/yuv444_10bit_rav1e.mp4")
    # plot_captured_data(in_fname="./captured_video/yuv420_10bit_svt.mp4")

    # GTX1060
    # plot_captured_data(
    #     in_fname="./captured_video/GTX1060_AVIF.mp4",
    #     graph_title="AVIF (10 bit, YUV444, Full Range, GTX 1060 Super)")
    # plot_captured_data(
    #     in_fname="./captured_video/GTX1060_YouTube.mp4",
    #     graph_title="YouTube (GTX 1060 Super)")
    # plot_captured_data(
    #     in_fname="./captured_video/GTX1060_MPC-BE_madVR.mp4",
    #     graph_title="MPC-BE with madVR (GTX 1060 Super)")
    # plot_captured_data(
    #     in_fname="./captured_video/GTX1060_Movies_and_TV_0.1-1000.mp4",
    #     graph_title="Movies & TV (GTX 1060 Super (0.1-1000 cd/m2))")
    # plot_captured_data(
    #     in_fname="./captured_video/GTX1060_Movies_and_TV_10-10000.mp4",
    #     graph_title="Movies & TV (GTX 1060 Super (10-10000 cd/m2))")
    # plot_captured_data(
    #     in_fname="./captured_video/GTX1060_YouTube.mp4",
    #     graph_title="YouTube (GTX 1060 Super)")
    # plot_captured_data(
    #     in_fname="./captured_video/GTX1060_VLC.mp4",
    #     graph_title="VLC (GTX 1060 Super)")

    # Ryzen
    # plot_captured_data(
    #     in_fname="./captured_video/Ryzen_4500U_AVIF.mp4",
    #     graph_title="AVIF (10 bit, YUV444, Full Range, RYZEN_4500U")
    # plot_captured_data(
    #     in_fname="./captured_video/Ryzen_4500U_VLC.mp4",
    #     graph_title="VLC (Ryzen 4500U)")
    # plot_captured_data(
    #     in_fname="./captured_video/Ryzen_4500U_Movies_and_TV_0.1-1000.mp4",
    #     graph_title="Movies & TV (Ryzen 4500U)")
    # plot_captured_data(
    #     in_fname="./captured_video/Ryzen_4500U_MPC-BE_madVR.mp4",
    #     graph_title="MPC-BE with madVR (Ryzen 4500U)")
    # plot_captured_data(
    #     in_fname="./captured_video/Ryzen_4500U_YouTube.mp4",
    #     graph_title="YouTube (Ryzen 4500U)")

    # Extra. changed the color accuracy mode on GTX1060 Super
    # plot_captured_data(
    #     in_fname="./captured_video/GTX1060_AVIF_Ref-Mode.mp4",
    #     graph_title="AVIF (GTX 1060 Super, Reference-mode)")
    # plot_captured_data(
    #     in_fname="./captured_video/GTX1060_MPC-BE_madVR_Ref-mode.mp4",
    #     graph_title="MPC-BE with madVR (GTX 1060 Super, Reference-mode)")
    # plot_captured_data(
    #     in_fname="./captured_video/GTX1060_AVIF_RenderingIntent-ABS.mp4",
    #     graph_title="AVIF (GTX 1060 Super, Rendering Intent-Absolute)")
    plot_captured_data(
        in_fname="./captured_video/GTX1060_MPC-BE_madVR_RI-ABS.mp4",
        graph_title="MPC-BE with madVR (GTX 1060 Super, RI-Absolute)")
