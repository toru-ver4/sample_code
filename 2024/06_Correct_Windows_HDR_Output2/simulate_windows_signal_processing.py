# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import normalised_primary_matrix
from colour.models import RGB_COLOURSPACE_BT709, RGB_COLOURSPACE_BT2020, \
    RGB_Colourspace
from scipy import linalg
import matplotlib.pyplot as plt


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

RED = np.array([255, 75, 0]) / 255  # for plot
GREEN = np.array([3, 175, 122]) / 255  # for plot
BLUE = np.array([0, 90, 255]) / 255  # for plot


def create_test_rgb_data(bit_depth=10):
    """
    Examples
    --------
    >>> rgb = create_test_rgb_data(bit_depth=10)
    >>> print(rgb)
    [[[   0    0    0]
      [   1    1    1]
      [   2    2    2]
      ...,
      [1021 1021 1021]
      [1022 1022 1022]
      [1023 1023 1023]]

     [[   0    0    0]
      [   1    0    0]
      [   2    0    0]
      ...,
      [1021    0    0]
      [1022    0    0]
      [1023    0    0]]

     [[   0    0    0]
      [   0    1    0]
      [   0    2    0]
      ...,
      [   0 1021    0]
      [   0 1022    0]
      [   0 1023    0]]

     ...,
     [[   0    0    0]
      [   1    0    1]
      [   2    0    2]
      ...,
      [1021    0 1021]
      [1022    0 1022]
      [1023    0 1023]]

     [[   0    0    0]
      [   1    1    0]
      [   2    2    0]
      ...,
      [1021 1021    0]
      [1022 1022    0]
      [1023 1023    0]]

     [[   0    0    0]
      [   0    1    1]
      [   0    2    2]
      ...,
      [   0 1021 1021]
      [   0 1022 1022]
      [   0 1023 1023]]]
    """
    num_of_cv = 2 ** bit_depth
    gradient = np.arange(num_of_cv, dtype=np.uint16)
    color_mask_list = np.array([
        [1, 1, 1],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 1], [1, 1, 0], [0, 1, 1]
    ], dtype=np.uint16)
    rgb = gradient.reshape(-1, num_of_cv, 1).repeat(3, axis=2)\
        .repeat(color_mask_list.shape[0], axis=0)

    rgb = rgb * color_mask_list.reshape(-1, 1, 3)

    return rgb


def st2084_eotf_fp32(x):
    x_fp32 = x.copy().astype(np.float32)
    m1 = np.float32(2610 / 16384)
    m2 = np.float32(2523 / 4096 * 128)
    c1 = np.float32(3424 / 4096)
    c2 = np.float32(2413 / 4096 * 32)
    c3 = np.float32(2392 / 4096 * 32)
    cc = np.float32(10000.0)

    div_m1 = np.float32(1.0) / m1
    div_m2 = np.float32(1.0) / m2

    numerator = np.maximum((x_fp32 ** div_m2) - c1, 0)

    y = (numerator / (c2 - c3 * (x_fp32 ** div_m2))) ** div_m1

    return y * cc


def st2084_inverse_eotf_fp32(x):
    x_fp32 = x.copy().astype(np.float32)
    x_fp32[x_fp32 < 0] = 0
    ll = x_fp32 / np.float32(10000.0)
    m1 = np.float32(2610 / 16384)
    m2 = np.float32(2523 / 4096 * 128)
    c1 = np.float32(3424 / 4096)
    c2 = np.float32(2413 / 4096 * 32)
    c3 = np.float32(2392 / 4096 * 32)

    y = ((c1 + c2 * (ll ** m1)) / (1 + c3 * (ll ** m1))) ** m2

    return y


def calc_rgb_to_rgb_matrix(
        src_color_space: RGB_Colourspace, dst_color_space: RGB_Colourspace):
    src_primaries = src_color_space.primaries.flatten()
    dst_primaries = dst_color_space.primaries.flatten()
    w = [0.3127, 0.3290]  # D65

    npm_src = normalised_primary_matrix(primaries=src_primaries, whitepoint=w)
    npm_dst = normalised_primary_matrix(primaries=dst_primaries, whitepoint=w)
    npm_dst_inv = linalg.inv(npm_dst)

    conv_mtx = npm_dst_inv.dot(npm_src)

    return conv_mtx


def apply_matrix_fp32(mtx, data):
    return np.einsum(
        "...ij,...j->...i", mtx, data, dtype=np.float32)


def calculate_convertion_matrix():
    rec709_to_rec2020_matrix = calc_rgb_to_rgb_matrix(
        src_color_space=RGB_COLOURSPACE_BT709,
        dst_color_space=RGB_COLOURSPACE_BT2020
    )
    rec709_to_rec2020_matrix_fp32 = rec709_to_rec2020_matrix.astype(np.float32)
    rec2020_to_rec709_matrix_fp32 = linalg.inv(rec709_to_rec2020_matrix_fp32)

    return rec709_to_rec2020_matrix_fp32, rec2020_to_rec709_matrix_fp32


def simulate_windows_signal_processing(enable_scrgb_fp32=False):
    reference_white_luminance = np.float32(100.0)

    # calculate convertion matrix
    rec709_to_rec2020_matrix_fp32, rec2020_to_rec709_matrix_fp32 =\
        calculate_convertion_matrix()

    # App simulation
    rec2100_pq_10bit = create_test_rgb_data()
    rec2100_pq_fp32 = rec2100_pq_10bit / np.float32(1023.0)
    rec2100_linear_fp32 = st2084_eotf_fp32(x=rec2100_pq_fp32)
    scrgb_fp32 = apply_matrix_fp32(
        mtx=rec2020_to_rec709_matrix_fp32, data=rec2100_linear_fp32
    )
    scrgb_fp32 = scrgb_fp32 / reference_white_luminance

    # DWM simulation
    if enable_scrgb_fp32:
        scrgb_fp16 = scrgb_fp32.copy()
    else:
        scrgb_fp16 = scrgb_fp32.astype(np.float16)

    # Display kernel simulation
    scrgb_fp32_disp = scrgb_fp16.copy().astype(np.float32)\
        * reference_white_luminance
    rec2100_linear_fp32_disp = apply_matrix_fp32(
        mtx=rec709_to_rec2020_matrix_fp32, data=scrgb_fp32_disp
    )
    rec2100_pq_fp32_disp = st2084_inverse_eotf_fp32(x=rec2100_linear_fp32_disp)
    rec2100_pq_10bit_disp = np.round(rec2100_pq_fp32_disp * 1023)\
        .astype(np.uint16)

    return rec2100_pq_10bit, scrgb_fp16, rec2100_pq_10bit_disp


def plot_result_st2084(data, title, graph_fname):
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]
    num_of_color = len(title_list)
    xx = np.arange(data.shape[1])
    _, axes = plt.subplots(num_of_color, 1, figsize=(8, 20))
    for idx in range(num_of_color):
        ax1 = axes[idx]
        yy = data[idx]
        ms = 4
        ax1.plot(xx, yy[..., 0], '-', ms=ms, color=RED, label="R")
        ax1.plot(xx, yy[..., 1], '-', ms=ms, color=GREEN, label="G")
        ax1.plot(xx, yy[..., 2], '-', ms=ms, color=BLUE, label="B")
        ax1.set_xticks([x * 128 for x in range(8)] + [1023])
        ax1.set_yticks([x * 256 for x in range(4)] + [1023])
        ax1.set_xlim([-10, 1033])
        ax1.set_ylim([-30, 1053])
        ax1.set_xlabel('Target Code Value (10-bit)')
        ax1.set_ylabel('Simulated Code Value (10-bit)')
        ax1.set_title(f'{title} - {title_list[idx]}')
        ax1.grid(True)
        ax1.legend(loc='upper left')

    plt.tight_layout()

    if graph_fname is not None:
        print(graph_fname)
        plt.savefig(graph_fname, dpi=100)
    else:
        plt.show()


def plot_result_scrgb(data, title, graph_fname):
    title_list = [
        "White", "Red", "Green", "Blue", "Majenta", "Yellow", "Cyan"
    ]
    num_of_color = len(title_list)
    xx = np.arange(data.shape[1])
    min_val = np.min(data)
    max_val = np.max(data)
    _, axes = plt.subplots(num_of_color, 1, figsize=(8, 20))
    for idx in range(num_of_color):
        ax1 = axes[idx]
        yy = data[idx]
        ms = 4
        ax1.plot(xx, yy[..., 0], '-', ms=ms, color=RED, label="R")
        ax1.plot(xx, yy[..., 1], '-', ms=ms, color=GREEN, label="G")
        ax1.plot(xx, yy[..., 2], '-', ms=ms, color=BLUE, label="B")
        ax1.set_xticks([x * 128 for x in range(8)] + [1023])
        ax1.set_xlim([-10, 1033])
        ax1.set_ylim([min_val*1.05, max_val*1.05])
        ax1.set_xlabel('Target Code Value (10-bit)')
        ax1.set_ylabel('scRGB (Simulated Value)')
        ax1.set_title(f'{title} - {title_list[idx]}')
        ax1.grid(True)
        ax1.legend(loc='upper left')

    plt.tight_layout()

    if graph_fname is not None:
        print(graph_fname)
        plt.savefig(graph_fname, dpi=100)
    else:
        plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_plot_different_bit_depth(bit_depth=16)
    # debug_plot_different_bit_depth(bit_depth=32)
    # debug_plot_different_bit_depth(bit_depth=64)
    # debug_matrix_error()
    # debug_1018cv()

    # data = scrgb_half_float_error_simulation(calc_dtype=np.float16)
    # plot_simulated_data(data=data)

    # dirext_x_app_ng_simulation(calc_dtype=np.float16)

    # DO NOT FORGET TO IMPLEMENT THIS FUNCTION!!!!!!
    # calc_matrix_based_on_DWM()

    # debug_plot_captured_three_tp(
    #     fname="./debug/capture/TP_Rec709_2020_17x17x17_Edge.png")
    # debug_plot_captured_three_tp(
    #     fname="./debug/capture/TP_Rec709_2020_17x17x17_MPC-BE.png")
    # calc_half_float_inv_rec709_to_rec2020_mtx()

    # plot_tp_10bit_green_high_luminance_hdmi()
    # create_tp_corrdinate_and_ref_value_csv()

    # plot_diff_rec709_rec2020_control()
    # create_diff_csv_17x17x17_control()

    # plot_inverse_st2084()

    # print(eotf_ST2084(128/1023))
    # x = np.array([0.01, 1000, 0.1])
    # y = np.round(tf.oetf_from_luminance(x, tf.ST2084) * 1023).astype(np.uint16)
    # print(y)

    rec2100_pq_10bit, scrgb_fp16, rec2100_pq_10bit_after =\
        simulate_windows_signal_processing(enable_scrgb_fp32=False)
    plot_result_st2084(
        data=rec2100_pq_10bit,
        title="Original Rec.2100-PQ",
        graph_fname="./debug/plot/simulated_data_src_rec2100_PQ.png"
    )
    plot_result_st2084(
        data=rec2100_pq_10bit_after,
        title="Simulated Rec.2100-PQ",
        graph_fname="./debug/plot/simulated_data_dst_rec2100_PQ.png"
    )
    plot_result_scrgb(
        data=scrgb_fp16,
        title="Simulated scRGB",
        graph_fname="./debug/plot/simulated_data_scRGB_rec2100_PQ.png"
    )

    # from colour.models import eotf_inverse_ST2084, eotf_ST2084
    # rgb = np.array([[0, 0, 0], [1, 16, 18], [768, 769, 770], [511, 512, 1023]], dtype=np.uint16)
    # y_fp64 = eotf_ST2084(rgb / 1023)
    # y_fp32 = st2084_eotf_fp32(rgb/np.float32(1023))
    # y_fp32 = y_fp64.astype(np.float32)
    # y_fp16 = y_fp64.astype(np.float16)

    # # print(y_fp64 - y_fp32)
    # # print(y_fp64 - y_fp16)

    # st2084_fp64 = np.round(eotf_inverse_ST2084(y_fp64) * 1023).astype(np.uint16)
    # st2084_fp16 = np.round(st2084_inverse_eotf_fp32(y_fp16) * 1023).astype(np.uint16)
    # print(st2084_fp64)
    # print(st2084_fp16)

    # print(RGB_COLOURSPACE_BT709.whitepoint)
