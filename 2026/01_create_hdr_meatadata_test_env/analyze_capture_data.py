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
import plot_utility as pu

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
COLOR_CHECKER_GM22_COLOR = np.clip(generate_color_checker_rgb_value(), 0.0, 1.0) ** (1/2.2)


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
    fig.savefig(output_graph_name, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def plot_all_capture_data():
    browser_list = ["Edge", "Chrome"]
    mhc_profile_list = ["BT.709-100nits", "BT.2020-10000nits"]
    sdr_content_brightness_list = ["SDR-80nits", "SDR-204nits"]
    # mhc_profile_list = ["BT.2020-10000nits"]
    # sdr_content_brightness_list = ["SDR-204nits"]

    for browser, mhc_profile, sdr_content_brightness in product(browser_list, mhc_profile_list, sdr_content_brightness_list):
        os_settings_dir = f"{browser}/{mhc_profile}_{sdr_content_brightness}/"

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


def compare_result_between_chrome_and_edge():
    browser_list = ["Edge", "Chrome"]
    mhc_profile_list = ["BT.709-100nits", "BT.2020-10000nits"]
    sdr_content_brightness_list = ["SDR-80nits", "SDR-204nits"]
    # mhc_profile_list = ["BT.2020-10000nits"]
    # sdr_content_brightness_list = ["SDR-204nits"]

    for mhc_profile, sdr_content_brightness in product(mhc_profile_list, sdr_content_brightness_list):

        mdcv_primaries_list = MDCV_PRIMARIES_LIST
        mdcv_luminance_list = MDCV_LUMINANCE_LIST
        clli_luminance_list = CLLI_LUMINANCE_LIST
        kind_ext_list = [
            # [KIND_AV1, ".mp4"],
            # [KIND_HEVC, ".mp4"],
            [KIND_AVIF, ".avif"],
            # [KIND_PNG, ".png"]
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
                compare_two_data(src_hdr_file_name, mhc_profile, sdr_content_brightness)
            else:
                continue


def compare_two_data(src_hdr_file_name, mhc_profile, sdr_content_brightness):
    def get_file_name(browser, src_hdr_file_name, mhc_profile, sdr_content_brightness):
        dir = f"{browser}/{mhc_profile}_{sdr_content_brightness}/"
        file_name = "./capture_img/" + dir + Path(src_hdr_file_name).stem + ".jxr"

        return file_name

    chrome_file_name = get_file_name("Chrome", src_hdr_file_name, mhc_profile, sdr_content_brightness)
    edge_file_name = get_file_name("Edge", src_hdr_file_name, mhc_profile, sdr_content_brightness)

    print(f"compare\n  {chrome_file_name}\n  {edge_file_name}")

    chrome_img = read_jxr_as_bt2020_linear(chrome_file_name)
    edge_img = read_jxr_as_bt2020_linear(edge_file_name)

    chrome_step_ramp_7colors_luminance = get_step_ramp_7colors(img=chrome_img, pos_list=STEP_RAMP_POS)
    edge_step_ramp_7colors_luminance = get_step_ramp_7colors(img=edge_img, pos_list=STEP_RAMP_POS)
        
    chrome_colorchecker_colors = get_colorchecker_colors(img=chrome_img, pos_list=COLORCHECKER_POS)
    edge_colorchecker_colors = get_colorchecker_colors(img=edge_img, pos_list=COLORCHECKER_POS)

    try:
        np.testing.assert_almost_equal(
            chrome_step_ramp_7colors_luminance,
            edge_step_ramp_7colors_luminance,
            decimal=7
        )
    except AssertionError as e:
        print(e)

    try:
        np.testing.assert_almost_equal(
            chrome_colorchecker_colors,
            edge_colorchecker_colors,
            decimal=7
        )
    except AssertionError as e:
        print(e)


def debug_check_two_data():
    chrome_fname = "./capture_img/Chrome/BT.2020-10000nits_SDR-204nits/avif_mdcv-p-None_mdcv-l-None_clli-100.jxr"
    edge_fname = "./capture_img/Edge/BT.2020-10000nits_SDR-204nits/avif_mdcv-p-None_mdcv-l-None_clli-100.jxr"

    # chrome_fname = "./capture_img/Chrome/BT.709-100nits_SDR-204nits/avif_mdcv-p-None_mdcv-l-None_clli-None.jxr"
    # edge_fname = "./capture_img/Edge/BT.709-100nits_SDR-204nits/avif_mdcv-p-None_mdcv-l-None_clli-None.jxr"

    chrome_img = read_jxr_as_bt2020_linear(chrome_fname)
    edge_img = read_jxr_as_bt2020_linear(edge_fname)

    chrome_step_ramp_7colors_luminance = get_step_ramp_7colors(img=chrome_img, pos_list=STEP_RAMP_POS)
    edge_step_ramp_7colors_luminance = get_step_ramp_7colors(img=edge_img, pos_list=STEP_RAMP_POS)

    chrome_colorchecker_colors = get_colorchecker_colors(img=chrome_img, pos_list=COLORCHECKER_POS)
    edge_colorchecker_colors = get_colorchecker_colors(img=edge_img, pos_list=COLORCHECKER_POS)

    diff = np.abs(chrome_step_ramp_7colors_luminance - edge_step_ramp_7colors_luminance)
    print(diff.shape)
    print(np.max(diff))
    print(np.unravel_index(np.argmax(diff), diff.shape))

    for idx in range(65):
        chrome = chrome_step_ramp_7colors_luminance[0, idx, 0]
        edge = edge_step_ramp_7colors_luminance[0, idx, 0]
        print(f"{idx}, {chrome:.3f}, {edge:.3f}")

    # np.testing.assert_almost_equal(
    #     chrome_step_ramp_7colors_luminance,
    #     edge_step_ramp_7colors_luminance,
    #     decimal=7
    # )

    # np.testing.assert_almost_equal(
    #     chrome_colorchecker_colors,
    #     edge_colorchecker_colors,
    #     decimal=7
    # )


def plot_gray_ramp_only(
        content_luminance, monitor_peak_luminance,
        sdr_80_fname, sdr_204_fname, output_fname):
    sdr_80_img = read_jxr_as_bt2020_linear(sdr_80_fname)
    sdr_204_img = read_jxr_as_bt2020_linear(sdr_204_fname)

    sdr_80_step_ramp = get_step_ramp_7colors(img=sdr_80_img, pos_list=STEP_RAMP_POS)[0, :, 1] * 100
    sdr_204_step_ramp = get_step_ramp_7colors(img=sdr_204_img, pos_list=STEP_RAMP_POS)[0, :, 1] * 100

    x_target = 10 ** np.linspace(0, 4, 65)

    fig, ax1 = pu.plot_1_graph(
        fontsize=18,
        figsize=(10, 6),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"HDR Content: {content_luminance} nits, Monitor: {monitor_peak_luminance} nits",
        graph_title_size=20,
        xlabel="Target Luminance (nits)",
        ylabel="Measured Luminance (nits)",
        axis_label_size=None,
        legend_size=18,
        xlim=[0.8, 12000],
        ylim=[0.8, 12000],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_scale_settings(
        ax1, grid_alpha=0.5, bg_color="#F0F0F0", grid_color="#808080", major_grid_color="#000000")
    ax1.plot(x_target, sdr_204_step_ramp, label="SDR content brightness = 204 nits")
    ax1.plot(x_target, sdr_80_step_ramp, label="SDR content brightness = 80 nits")
    print(output_fname)
    pu.show_and_save(fig=fig, legend_loc='upper left', save_fname=output_fname, show=False)


def plot_gray_ramp_for_blog(
        content_luminance, monitor_peak_luminance,
        sdr_204_fname, output_fname):
    sdr_204_img = read_jxr_as_bt2020_linear(sdr_204_fname)

    sdr_204_step_ramp = get_step_ramp_7colors(img=sdr_204_img, pos_list=STEP_RAMP_POS)[0, :, 1] * 100

    x_target = 10 ** np.linspace(0, 4, 65)

    fig, ax1 = pu.plot_1_graph(
        fontsize=18,
        figsize=(10, 6),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"HDR10 Metadata: {content_luminance} nits, Monitor: {monitor_peak_luminance} nits",
        graph_title_size=22,
        xlabel="Target Luminance (nits)",
        ylabel="Measured Luminance (nits)",
        axis_label_size=None,
        legend_size=18,
        xlim=[0.8, 12000],
        ylim=[0.8, 12000],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=4,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_scale_settings(
        ax1, grid_alpha=0.5, bg_color="#F0F0F0", grid_color="#808080", major_grid_color="#000000")
    ax1.plot(x_target, sdr_204_step_ramp, color=pu.RED, label="Browser Display")
    # ax1.plot(x_target, sdr_80_step_ramp, label="SDR content brightness = 80 nits")
    ax1.plot(x_target, x_target, '--k', lw=1.5, label="Original")
    print(output_fname)
    pu.show_and_save(fig=fig, legend_loc='upper left', save_fname=output_fname, show=False)


def plot_specific_param_ramp_pattern():
    browser_list = ["Edge"]
    mhc_profile_list = ["BT.709-100nits", "BT.2020-10000nits"]

    for browser, mhc_profile in product(browser_list, mhc_profile_list):
        mdcv_primaries_list = [None]
        mdcv_luminance_list = [None]
        clli_luminance_list = CLLI_LUMINANCE_LIST
        kind_ext_list = [
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
                def make_capture_fname(sdr_content_brightness):
                    os_settings_dir = f"{browser}/{mhc_profile}_{sdr_content_brightness}/"
                    capture_file_name = "./capture_img/" + os_settings_dir + Path(src_hdr_file_name).stem + ".jxr"

                    return capture_file_name

                sdr_80nits_fname = make_capture_fname("SDR-80nits")
                sdr_204nits_fname = make_capture_fname("SDR-204nits")
                output_graph_name = f"./img/{Path(sdr_80nits_fname).stem}_{mhc_profile}.png"
                output_graph_name_blog = f"./img/{Path(sdr_80nits_fname).stem}_{mhc_profile}_blog.png"
                monitor_luminance = mhc_profile.split("-")[1].replace("nits", "")
                # plot_gray_ramp_only(
                #     content_luminance=clli_luminance, monitor_peak_luminance=monitor_luminance,
                #     sdr_80_fname=sdr_80nits_fname, sdr_204_fname=sdr_204nits_fname, output_fname=output_graph_name
                # )
                plot_gray_ramp_for_blog(
                    content_luminance=clli_luminance, monitor_peak_luminance=monitor_luminance,
                    sdr_204_fname=sdr_204nits_fname, output_fname=output_graph_name_blog
                )
                # break
            else:
                continue
        # break


def analyze_auto_capture_data():
    pass
    # plot_all_capture_data()
    # compare_result_between_chrome_and_edge()
    plot_specific_param_ramp_pattern()


def read_ramp_data_from_jxr(jxr_file: str):
    img = read_jxr_as_bt2020_linear(jxr_file)
    step_ramp_7colors_luminance = get_step_ramp_7colors(img=img, pos_list=STEP_RAMP_POS) * 100    

    return step_ramp_7colors_luminance[0, :, 1]


def analyze_no_metadata_capture_data():
    luminance_list = [
        200, 400, 600, 700, 800, 900, 1000, 4000, 10000
    ]
    fname_list = [
        f"./capture_img/No_Metadata/monitor-{luminance:05d}.jxr" for luminance in luminance_list
    ]
    data = np.zeros((len(luminance_list), 65))
    for idx, fname in enumerate(fname_list):
        ramp = read_ramp_data_from_jxr(jxr_file=fname)
        data[idx] = ramp

    x_target = 10 ** np.linspace(0, 4, 65)

    fig, ax1 = pu.plot_1_graph(
        fontsize=14,
        figsize=(10, 6),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Tone Mapping for PNG File Without Metadata",
        graph_title_size=22,
        xlabel="Target Luminance (nits)",
        ylabel="Measured Luminance (nits)",
        axis_label_size=None,
        legend_size=13,
        xlim=[8, 12000],
        ylim=[5, 12000],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_scale_settings(
        ax1, grid_alpha=0.5, bg_color="#F0F0F0", grid_color="#808080", major_grid_color="#000000")
    for idx, luminance in enumerate(luminance_list):
        ax1.plot(x_target, data[idx], label=f"Monitor: {luminance} nits")
    ax1.plot(x_target, x_target, '--k', lw=1.5, label="Reference")
    output_fname = "./img/no_metadata_comparison.png"
    print(output_fname)
    pu.show_and_save(fig=fig, legend_loc='upper left', save_fname=output_fname, show=True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # analyze_auto_capture_data()
    analyze_no_metadata_capture_data()
    # debug_check_two_data()

    # plot_gray_ramp_for_blog(
    #     content_luminance=10000, monitor_peak_luminance=100,
    #     sdr_204_fname="./capture_img/No_Metadata/monitor_100-100.jxr",
    #     output_fname="./img/100-100.png"
    # )
