# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from itertools import product

# import third-party libraries
import numpy as np
from colour import read_image

# import my libraries
import test_pattern_generator2 as tpg
import ty_davinci_control_lib as dcl
from davinci17_cms_explore import SDR_CLIP_NAME, HDR_CLIP_NAME, EXR_CLIP_NAME,\
    make_output_path, MEDIA_DST_PATH, EXR_MIN_EXPOSURE, EXR_MAX_EXPOSURE,\
    get_media_src_fname_exr
import plot_utility as pu
import transfer_functions as tf
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


RCM_PRESET_COLOR_SPACE_TO_GAMUT = {
    dcl.RCM_PRESET_SDR_709: cs.BT709,
    dcl.RCM_PRESET_SDR_2020: cs.BT2020,
    dcl.RCM_PRESET_SDR_2020_P3_LIMITED: cs.BT2020,
    dcl.RCM_PRESET_SDR_P3_D60: None,
    dcl.RCM_PRESET_HDR_DAVINCI_INTERMEDIATE: None,
    dcl.RCM_PRESET_HDR_2020_INTERMEDIATE: cs.BT2020,
    dcl.RCM_PRESET_HDR_2020_HLG: cs.BT2020,
    dcl.RCM_PRESET_HDR_2020_HLG_P3_LIMITED: cs.BT2020,
    dcl.RCM_PRESET_HDR_2020_PQ: cs.BT2020,
    dcl.RCM_PRESET_HDR_2020_PQ_P3_LIMITED: cs.BT2020,
    dcl.RCM_PRESET_CUSTOM: None
}

RCM_COLOR_SPACE_TO_GAMUT = {
    dcl.RCM_COLOR_SPACE_709_GM24: cs.BT709,
    dcl.RCM_COLOR_SPACE_2020_GM24: cs.BT2020,
    dcl.RCM_COLOR_SPACE_2020_ST2084: cs.BT2020,
    dcl.RCM_COLOR_SPACE_LINER: None,
}

RCM_COLOR_SPACE_TO_GAMMA = {
    dcl.RCM_COLOR_SPACE_709_GM24: tf.GAMMA24,
    dcl.RCM_COLOR_SPACE_2020_GM24: tf.GAMMA24,
    dcl.RCM_COLOR_SPACE_2020_ST2084: tf.ST2084,
    dcl.RCM_COLOR_SPACE_LINER: None
}


def get_gamut_from_rcm_color_space(
        rcm_color_space, rcm_preset_color_space=dcl.RCM_PRESET_SDR_709):
    print(f"rcm_color_space = {rcm_color_space}")
    gamut = RCM_COLOR_SPACE_TO_GAMUT[rcm_color_space]

    if gamut is None:
        gamut = RCM_PRESET_COLOR_SPACE_TO_GAMUT[rcm_preset_color_space]

    return gamut


def calc_result_sdr(file_path, in_color_space, out_color_space):
    # print(f"SDR: {file_path}")
    pass


def calc_result_hdr(file_path, in_color_space, out_color_space):
    # print(f"HDR: {file_path}")
    pass


def calc_result_exr(
        file_path, in_color_space, working_color_space, out_color_space):
    print(f"EXR: {file_path}")
    exr_in_out_characteristics(
        file_path=file_path, in_color_space=in_color_space,
        working_color_space=working_color_space,
        out_color_space=out_color_space)


def exr_in_out_characteristics(
        file_path, in_color_space, working_color_space, out_color_space):
    """
    Parameters
    ----------
    file_path : str
        file path of the DaVinci Resolve output
    """

    # gray scale
    in_gamut = get_gamut_from_rcm_color_space(
        rcm_color_space=in_color_space,
        rcm_preset_color_space=working_color_space)
    if in_gamut is None:
        print("Error. Analyzation is failed.")
        return

    out_img = read_image(path=file_path)
    out_line_g = out_img[0, :, 1]
    tf_str = RCM_COLOR_SPACE_TO_GAMMA[out_color_space]
    out_gray_spec = tf.eotf_to_luminance(out_line_g, tf_str)
    out_spec = out_gray_spec / 100

    in_img_name = get_media_src_fname_exr(idx=0)
    in_img = read_image(path=in_img_name)
    in_spec = in_img[0, :, 1]

    graph_title = f"Linear --> {working_color_space} --> {out_color_space}"

    fig, ax1 = pu.plot_1_graph(
        fontsize=18,
        figsize=(12, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=graph_title,
        graph_title_size=16,
        xlabel="Linear (1.0 = 100 nits)",
        ylabel="Linear (1.0 = 100 nits)",
        axis_label_size=None,
        legend_size=17,
        xlim=[3.5481338829296651e-07, 2818.38293162024],
        ylim=[3.9642404597707485e-07, 251.23932593436777],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    pu.log_scale_settings_exr(ax1)
    ax1.plot(in_spec, out_spec, label="EXR In-Out")
    # print(f"x_lim = {ax1.get_xlim()}")
    # print(f"y_lim = {ax1.get_ylim()}")
    pu.show_and_save(
        fig=fig, legend_loc=None, save_fname="./img/sample.png")


def make_file_path(clip_name, color_process_mode, out_color_space):
    output_path = make_output_path(
        src_name=clip_name, out_dir=MEDIA_DST_PATH,
        processing_mode=color_process_mode,
        output_color_space=out_color_space)
    return str(output_path) + '_00086400.tif'


def analyze_main():
    clip_name_list = [SDR_CLIP_NAME, HDR_CLIP_NAME, EXR_CLIP_NAME]
    clip_color_space_list = [
        dcl.RCM_COLOR_SPACE_709_GM24, dcl.RCM_COLOR_SPACE_2020_ST2084,
        dcl.RCM_COLOR_SPACE_LINER]
    clip_pair_list = [
        [a, b] for a, b in zip(clip_name_list, clip_color_space_list)]
    color_process_mode_list = [
        dcl.RCM_PRESET_SDR_709,
        dcl.RCM_PRESET_SDR_2020,
        dcl.RCM_PRESET_SDR_2020_P3_LIMITED,
        dcl.RCM_PRESET_SDR_P3_D60,
        dcl.RCM_PRESET_HDR_DAVINCI_INTERMEDIATE,
        dcl.RCM_PRESET_HDR_2020_INTERMEDIATE,
        dcl.RCM_PRESET_HDR_2020_HLG,
        dcl.RCM_PRESET_HDR_2020_HLG_P3_LIMITED,
        dcl.RCM_PRESET_HDR_2020_PQ,
        dcl.RCM_PRESET_HDR_2020_PQ_P3_LIMITED
    ]
    out_color_space_list = [
        dcl.RCM_COLOR_SPACE_709_GM24,
        dcl.RCM_COLOR_SPACE_2020_ST2084]

    for clip_pair, color_process_mode, out_color_space in product(
            clip_pair_list, color_process_mode_list, out_color_space_list):
        clip_name = clip_pair[0]
        clip_color_space = clip_pair[1]

        file_path = make_file_path(
            clip_name=clip_name, color_process_mode=color_process_mode,
            out_color_space=out_color_space)

        if 'dst_sdr' in file_path:
            calc_result_sdr(
                file_path=file_path, in_color_space=clip_color_space,
                out_color_space=out_color_space)
            # break
        elif 'dst_hdr' in file_path:
            calc_result_hdr(
                file_path=file_path, in_color_space=clip_color_space,
                out_color_space=out_color_space)
            # break
        elif 'dst_exr' in file_path:
            calc_result_exr(
                file_path=file_path, in_color_space=clip_color_space,
                working_color_space=color_process_mode,
                out_color_space=out_color_space)
            # break


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    analyze_main()
