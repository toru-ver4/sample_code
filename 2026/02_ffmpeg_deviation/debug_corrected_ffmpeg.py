# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
import shlex
import subprocess
from pathlib import Path

# import third-party libraries
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# import my libraries
import test_pattern_generator2 as tpg
import color_space as cs

TARGET_DIR = (Path(__file__).resolve().parent.parent.parent / "2026" / "01_create_hdr_meatadata_test_env").resolve()
if str(TARGET_DIR) not in sys.path:
    sys.path.insert(0, str(TARGET_DIR))

from analyze_adaptive_hdr_tp import (
    read_jxr_as_bt2020_linear,
    get_step_ramp_pos_list,
    get_step_ramp_7colors,
    get_colorchecker_pos_list,
    get_colorchecker_colors,
    calc_bt2020_colorchecker_de2000,
)

STEP_RAMP_POS = get_step_ramp_pos_list(img_width=1920, img_height=1080)
COLORCHECKER_POS = get_colorchecker_pos_list(img_width=1920, img_height=1080)
COLOR_CHECKER_GM22_COLOR = np.clip(tpg.generate_color_checker_rgb_value(), 0.0, 1.0) ** (1/2.2)


def encode_using_ffmpeg(ffmpeg_type="original"):
    if ffmpeg_type == 'original':
        cmd = 'ffmpeg'
        output_name = "./debug/1920x1080_ST2084_Rec.2020_original.mp4"
    elif ffmpeg_type == 'corrected':
        cmd = "/opt/my_ffmpeg_out/bin/ffmpeg"
        output_name = "./debug/1920x1080_ST2084_Rec.2020_corrected.mp4"
    else:
        raise ValueError("Invalid ffmpeg_type")
    
    src_png_name = "./img/1920x1080_ST2084_Rec.2020.png"
    ops = [
        '-hide_banner',
        '-loop', '1',
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-framerate', '24',
        '-t', "5",
        '-i', src_png_name,
        '-c:v', 'libx265',
        '-pix_fmt', 'yuv420p10le',
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-qp', '0',
        output_name,
        '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def read_jxr_as_bt2020_linear(fname: str):
    scrgb_img = tpg.img_read_as_float(fname)
    bt2020_img = cs.rgb_to_rgb(scrgb_img, cs.BT709, cs.BT2020)

    return bt2020_img


def decode_and_plot(ffmpeg_type="original"):
    if ffmpeg_type == 'original':
        src_fname = "./debug/1920x1080_ST2084_Rec.2020_original.jxr"
        graph_title = "Original FFmpeg"
        graph_fname = "./img/1920x1080_ST2084_Rec.2020_original_plot.png"

    elif ffmpeg_type == 'corrected':
        src_fname = "./debug/1920x1080_ST2084_Rec.2020_corrected.jxr"
        graph_title = "Corrected FFmpeg"
        graph_fname = "./img/1920x1080_ST2084_Rec.2020_corrected_plot.png"
    else:
        raise ValueError("Invalid ffmpeg_type")
    
    img = read_jxr_as_bt2020_linear(src_fname)
    step_ramp_7colors_luminance = get_step_ramp_7colors(img=img, pos_list=STEP_RAMP_POS) * 100
    colorchecker_colors = get_colorchecker_colors(img=img, pos_list=COLORCHECKER_POS)
    cc_de2000 = calc_bt2020_colorchecker_de2000(rgb_linear=colorchecker_colors)
    cc_plot_colors = COLOR_CHECKER_GM22_COLOR

    color_names = ["White", "Yellow", "Cyan", "Green", "Magenta", "Red", "Blue"]
    channel_names = ["R", "G", "B"]
    channel_colors = ["#FF0000", "#00AA00", "#0000FF"]
    major_ticks = [1, 10, 100, 1000, 10000]
    minor_ticks = [
        2, 3, 4, 5, 6, 7, 8, 9,
        20, 30, 40, 50, 60, 70, 80, 90,
        200, 300, 400, 500, 600, 700, 800, 900,
        2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    ]

    x_target = 10 ** np.linspace(0, 4, 65)
    x_ref = np.linspace(1.0, 10000.0, 1024)
    y_ref = x_ref

    fig, axes = plt.subplots(
        nrows=8, ncols=1, figsize=(12, 20)
    )
    axes = np.atleast_1d(axes)

    cc_ax = axes[0]
    cc_x = np.arange(1, len(cc_de2000) + 1)
    cc_ax.bar(
        cc_x,
        cc_de2000,
        color=cc_plot_colors,
        edgecolor='black',
        linewidth=0.8
    )
    cc_ax.set_xlim(0.25, len(cc_de2000) + 0.75)
    cc_ax.set_ylim(0.0, 30.0)
    cc_ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    cc_ax.set_ylabel("CIE DE2000")
    cc_ax.set_xticks([])
    cc_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    cc_ax.grid(axis='y', color='gray', linestyle='-', linewidth=0.8)

    for color_idx, ax in enumerate(axes[1:]):
        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)
        ax.set_xlim(1.0, 10000.0)
        ax.set_ylim(1.0, 10000.0)

        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)

        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.8)
        ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)

        ax.plot(x_ref, y_ref, linestyle='--', color='gray', linewidth=1.0)

        for ch_idx, (channel_name, line_color) in enumerate(zip(channel_names, channel_colors)):
            ax.plot(
                x_target,
                step_ramp_7colors_luminance[color_idx, :, ch_idx],
                color=line_color,
                linewidth=1.5,
                label=f"{color_names[color_idx]} - {channel_name}"
            )

        ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(graph_fname, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def save_as_exr(ffmpeg_type="original"):
    if ffmpeg_type == 'original':
        src_fname = "./debug/1920x1080_ST2084_Rec.2020_original.jxr"
    elif ffmpeg_type == 'corrected':
        src_fname = "./debug/1920x1080_ST2084_Rec.2020_corrected.jxr"
    else:
        raise ValueError("Invalid ffmpeg_type")
    
    tpg.jxr_to_exr(src_fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # encode_using_ffmpeg(ffmpeg_type='original')
    # decode_and_plot(ffmpeg_type="original")
    save_as_exr(ffmpeg_type="original")

    # encode_using_ffmpeg(ffmpeg_type='corrected')
    # decode_and_plot(ffmpeg_type="corrected")
    save_as_exr(ffmpeg_type="corrected")
