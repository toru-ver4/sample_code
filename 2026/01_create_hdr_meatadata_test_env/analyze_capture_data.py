import os
import sys
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from colour.difference import delta_E_CIE2000
from colour import XYZ_to_Lab

from test_pattern_generator2 import img_read_as_float, get_colorchecker_ref_lab, generate_color_checker_rgb_value
import color_space as cs
from analyze_adaptive_hdr_tp import (
    read_jxr_as_bt2020_linear,
    get_step_ramp_pos_list,
    get_step_ramp_7colors,
    get_colorchecker_pos_list,
    get_colorchecker_colors,
    calc_bt2020_colorchecker_de2000,
)

TARGET_DIR = (Path(__file__).resolve().parent.parent.parent / "2025" / "12_AVIF_UltraHDR_PNG_Comparison").resolve()
if str(TARGET_DIR) not in sys.path:
    sys.path.insert(0, str(TARGET_DIR))

from create_hdr_media import (  # noqa: E402
    MDCV_PRIMARIES_LIST,
    MDCV_LUMINANCE_LIST,
    CLLI_LUMINANCE_LIST,
    KIND_AV1,
    KIND_AVIF,
    KIND_HEVC,
    KIND_PNG,
    make_media_file_name_without_ext
)


STEP_RAMP_POS = get_step_ramp_pos_list(img_width=1920, img_height=1080)
COLORCHECKER_POS = get_colorchecker_pos_list(img_width=1920, img_height=1080)
COLOR_CHECKER_GM22_COLOR = generate_color_checker_rgb_value() ** (1/2.2)


def plot_all_data(capture_file_name: str, output_graph_dir: str):
    output_graph_name = str(Path(output_graph_dir) / Path(Path(capture_file_name).stem + ".png"))

    img = read_jxr_as_bt2020_linear(capture_file_name)
    step_ramp_7colors_luminance = get_step_ramp_7colors(img=img, pos_list=STEP_RAMP_POS) * 100
    colorchecker_colors = get_colorchecker_colors(img=img, pos_list=COLORCHECKER_POS)
    cc_de2000 = calc_bt2020_colorchecker_de2000(rgb_linear=colorchecker_colors)
    cc_plot_colors = COLOR_CHECKER_GM22_COLOR

    Path(output_graph_dir).mkdir(parents=True, exist_ok=True)

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
    cc_ax.set_ylim(0.0, 10.0)
    cc_ax.set_yticks([0, 2, 4, 6, 8, 10])
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
    fig.savefig(output_graph_name, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def create_param_list():
    mhc_profile_list = ["BT.709-100nits", "BT.2020-10000nits"]
    sdr_content_brightness_list = ["SDR-80nits", "SDR-204nits"]
    # mhc_profile_list = ["BT.2020-10000nits"]
    # sdr_content_brightness_list = ["SDR-204nits"]

    for mhc_profile, sdr_content_brightness in product(mhc_profile_list, sdr_content_brightness_list):
        os_settings_dir = f"{mhc_profile}_{sdr_content_brightness}/"

        mdcv_primaries_list = MDCV_PRIMARIES_LIST
        mdcv_luminance_list = MDCV_LUMINANCE_LIST
        clli_luminance_list = CLLI_LUMINANCE_LIST
        kind_ext_list = [
            [KIND_AV1, ".mp4"],
            [KIND_HEVC, ".mp4"],
            [KIND_AVIF, ".avif"],
            [KIND_PNG, ".png"]
        ]
        for mdcv_primaries, mdcv_luminance, clli_luminance, (kind, ext)\
            in product(mdcv_primaries_list, mdcv_luminance_list, clli_luminance_list, kind_ext_list):
            if (mdcv_primaries is None) and (mdcv_luminance is not None):
                continue
            if (mdcv_primaries is not None) and (mdcv_luminance is None):
                continue
            file_name_without_ext = make_media_file_name_without_ext(
                kind=kind,
                suffix=None,
                mdcv_primaries=mdcv_primaries,
                mdcv_luminance=mdcv_luminance,
                clli_luminance=clli_luminance,
                dst_dir="../../2025/12_AVIF_UltraHDR_PNG_Comparison/hdr_media"
            )
            src_hdr_file_name = file_name_without_ext + ext
            if os.path.exists(src_hdr_file_name):
                capture_file_name = "./capture_img/" + os_settings_dir + Path(src_hdr_file_name).stem + ".jxr"
                output_graph_dir = "./graph_img/" + os_settings_dir
                print(capture_file_name)
                plot_all_data(capture_file_name=capture_file_name, output_graph_dir=output_graph_dir)
                # break
            else:
                continue
        # break


def plot_colorchecker_and_step_ramp_7color(img_fname:str):
    graph_fname = f"./graph_img/{Path(img_fname).stem}"
    print(graph_fname)


def analyze_capture_data():
    create_param_list()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    analyze_capture_data()
