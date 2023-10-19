# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import xy_to_xyY, xyY_to_XYZ, LUT3D, write_LUT
from colour.io import write_image, read_image
from colour.utilities import tstack
from colour.models import eotf_sRGB, log_encoding_ARRILogC4, \
    log_encoding_SLog3, oetf_ARIBSTDB67, eotf_inverse_ST2084, \
    oetf_inverse_ARIBSTDB67

# import my libraries
import color_space as cs
import test_pattern_generator2 as tpg
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


EXP_RANGE = 12
EXP_REF_VAL = 0.18


def log_encoding_apple_log(x):
    """
    Scence linear to Apple Log.

    Parameters
    ----------
    x : ndarray
        A scene linear light (1.0 is SDR diffuse white)

    Examples
    --------
    >>> log_encoding_apple_log(0.18)
    0.488271418052

    Note
    ----
    Valid range of `x` is from 2^-12 to 2^12
    """
    lut_1d = np.load("./secret/apple_log_encode_lut.npy")
    y = np.interp(x, xp=lut_1d[..., 0], fp=lut_1d[..., 1])

    return y


def log_decoding_apple_log(x):
    """
    Scence Apple Log to linear.

    Parameters
    ----------
    x : ndarray
        Apple Log value

    Examples
    --------
    >>> log_decoding_apple_log(0.488271418052)
    0.179995837368
    """
    lut_1d = np.load("./secret/apple_log_decode_lut.npy")
    y = np.interp(x, xp=lut_1d[..., 0], fp=lut_1d[..., 1])

    return y


def create_test_pattern_for_verify_on_Resolve():
    width = 1920
    height = 1080
    ref_val = 0.18
    exp_range = 6

    # apply oetf
    x_scene = np.linspace(
        ref_val*(2**(-exp_range)), ref_val*(2**(exp_range)), width)
    apple_log = log_encoding_apple_log(x_scene)
    apple_log_img = tpg.h_mono_line_to_img(apple_log, height)
    fname = f"./img/verify_apple_log_src-{exp_range}_to_{exp_range}_stops.png"
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, apple_log_img)

    # apply eotf
    x_apple_log = np.linspace(0, 1, width)
    scene_linear = log_decoding_apple_log(x_apple_log)
    apple_linear_img = tpg.h_mono_line_to_img(scene_linear, height)
    fname = "./img/verify_scene_linear_src-0_to_1023_apple_log.exr"
    print(fname)
    write_image(apple_linear_img, fname)


def create_test_pattern_for_apple_log_encoding_analysis():
    width = 1920
    height = 1080

    ref_val = EXP_REF_VAL
    exp_range = EXP_RANGE
    x = tpg.get_log2_x_scale(
        sample_num=width, ref_val=ref_val,
        min_exposure=-exp_range, max_exposure=exp_range)
    img = tpg.h_mono_line_to_img(x, height)
    fname = f"./img/src_log2_-{exp_range}_to_{exp_range}_stops.exr"
    print(fname)
    write_image(img, fname)


def create_test_pattern_for_apple_log_decoding_analysis():
    width = 1920
    height = 1080

    x = np.linspace(0, 1, width)
    img = tpg.h_mono_line_to_img(x, height)
    fname = "./img/src_0_to_1.png"
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, img)


def debug_plot_apple_log_encoding():
    fname = "./secret/apple_log_encode_out.exr"
    img = read_image(fname)
    width = img.shape[1]
    ref_val = EXP_REF_VAL
    exp_range = EXP_RANGE
    x_resolve = tpg.get_log2_x_scale(
        sample_num=width, ref_val=ref_val,
        min_exposure=-exp_range, max_exposure=exp_range)
    y_resolve = img[0, ..., 1]
    x_lut = tpg.get_log2_x_scale(
        sample_num=36, ref_val=ref_val,
        min_exposure=-exp_range, max_exposure=exp_range)
    y_lut = log_encoding_apple_log(x_lut)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Apple Log Encoding Characteristics",
        xlabel=r"Stops from 18% gray",
        ylabel="Code Value",
        axis_label_size=None,
        ylim=[-0.05, 1.65],
        ytick=[x * 0.2 for x in range(9)],
        legend_size=17,
        linewidth=3)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks([2**i for i in range(-exp_range, exp_range+1, 2)])
    ax1.set_xticklabels([str(i) for i in range(-exp_range, exp_range+1, 2)])
    ax1.plot(
        x_resolve/ref_val, y_resolve, color=pu.RED, label="Apple_Log_EXR_Out")
    ax1.plot(
        x_lut/ref_val, y_lut, '--o', color=pu.GREEN, lw=1,
        label="Apple_Log_1DLUT")
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=True,
        save_fname="./img/apple_log_encoding_characteristics.png")


def debug_plot_apple_log_decoding():
    fname = "./secret/apple_log_decode_out.exr"
    img = read_image(fname)
    width = img.shape[1]
    x_resolve = np.linspace(0, 1, width)
    y_resolve = img[0, ..., 1]
    x_lut = np.linspace(0, 1, 256)
    y_lut = log_decoding_apple_log(x_lut)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Apple Log Encoding Characteristics",
        xlabel="Code Value (10-bit)",
        ylabel="Scene Linear Value",
        axis_label_size=None,
        legend_size=17,
        linewidth=3)
    ax1.set_yscale('log', base=2)
    ax1.plot(
        x_resolve, y_resolve, color=pu.RED, label="Apple_Log_EXR_Out")
    ax1.plot(
        x_lut, y_lut, '--', color=pu.GREEN, lw=1,
        label="Apple_Log_1DLUT")
    pu.show_and_save(
        fig=fig, legend_loc='upper left', show=True,
        save_fname="./img/apple_log_decoding_characteristics.png")


def save_apple_log_as_1dlut():
    # encoding function
    fname = "./secret/apple_log_encode_out.exr"
    img = read_image(fname)
    width = img.shape[1]
    ref_val = EXP_REF_VAL
    exp_range = EXP_RANGE

    x = tpg.get_log2_x_scale(
        sample_num=width, ref_val=ref_val,
        min_exposure=-exp_range, max_exposure=exp_range)
    y = img[0, ..., 1]
    lut = tstack([x, y])
    np.save("./secret/apple_log_encode_lut.npy", lut)

    # deconding function
    fname = "./secret/apple_log_decode_out.exr"
    img = read_image(fname)
    width = img.shape[1]
    x = np.linspace(0, 1, width)
    y = img[0, ..., 1]
    lut = tstack([x, y])
    np.save("./secret/apple_log_decode_lut.npy", lut)


def create_el_zone_3dlut(
        decode_func=log_decoding_apple_log, fname_prefix="Apple_Log"):
    """
    A test implementation of EL Zone System.
    https://www.elzonesystem.com/
    """
    def rgb_to_y_bt2020(rgb):
        y = rgb[..., 0] * 0.2627 + rgb[..., 1] * 0.6780 + rgb[..., 2] * 0.0593
        return y

    gray18 = 0.18
    lut3d_num_of_grid = 65
    exp_range = 7
    stops_list = [
        -7, -6, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5, 6, 7]
    color_list_8bit = [
        [3, 3, 3], [98, 71, 155], [158, 126, 184], [24, 116, 167],
        [39, 174, 228], [27, 168, 75], [93, 187, 71], [148, 200, 64],
        [144, 140, 135], [251, 232, 0], [255, 248, 166], [244, 112, 42],
        [247, 170, 71], [239, 28, 38], [229, 126, 140], [243, 190, 192],
        [255, 255, 255]
    ]
    color_list_linear = eotf_sRGB(np.array(color_list_8bit) / 255)
    color_list_log = color_list_linear ** (1/2.4)

    log_value = LUT3D.linear_table(lut3d_num_of_grid)
    out_lut = np.zeros_like(log_value)
    linear = decode_func(log_value)
    linear_y = rgb_to_y_bt2020(linear)

    for idx, stops in enumerate(stops_list):
        if stops == -exp_range:
            upper_diff = stops_list[idx + 1] - stops_list[idx]
            high_stops = stops + upper_diff / 2
            low_stops = -20
        elif stops == exp_range:
            lower_diff = stops_list[idx] - stops_list[idx - 1]
            high_stops = 20
            low_stops = stops - lower_diff / 2
        else:
            upper_diff = stops_list[idx + 1] - stops_list[idx]
            lower_diff = stops_list[idx] - stops_list[idx - 1]
            high_stops = stops + upper_diff / 2
            low_stops = stops - lower_diff / 2
        low_value = gray18 * (2 ** low_stops)
        high_value = gray18 * (2 ** high_stops)
        # print(f"{stops}, {low_stops}, {high_stops}, {low_value}, {high_value}")

        t_idx = ((low_value <= linear_y) & (linear_y < high_value))
        out_lut[t_idx] = color_list_log[idx]

    info_txt = f"EL Zone System for {fname_prefix}"
    lut_fname = f"./lut/EL_Zone_System_for_{fname_prefix}_"
    lut_fname += f"{lut3d_num_of_grid}-grid_BT1886.cube"
    lut3d = LUT3D(table=out_lut, name=info_txt)

    write_LUT(lut3d, lut_fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_test_pattern_for_apple_log_encoding_analysis()
    # create_test_pattern_for_apple_log_decoding_analysis()
    # create_test_pattern_for_verify_on_Resolve()
    # save_apple_log_as_1dlut()
    # debug_plot_apple_log_encoding()
    # debug_plot_apple_log_decoding()

    create_el_zone_3dlut(
        decode_func=oetf_inverse_ARIBSTDB67, fname_prefix="HLG")
    create_el_zone_3dlut(
        decode_func=log_decoding_apple_log, fname_prefix="Apple_Log")
