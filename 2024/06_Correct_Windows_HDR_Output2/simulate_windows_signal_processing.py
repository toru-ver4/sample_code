# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np
from colour.algebra import vector_dot
from colour import normalised_primary_matrix
from scipy import linalg
import matplotlib.pyplot as plt

# import my libraries
import test_pattern_generator2 as tpg
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_test_rgb_data():
    """
    Examples
    --------
    >>> rgb = create_test_rgb_data()
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
    bit_depth = 10
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
    m1 = np.float32(2610 / 16384)
    m2 = np.float32(2523 / 4096 * 128)
    c1 = np.float32(3424 / 4096)
    c2 = np.float32(2413 / 4096 * 32)
    c3 = np.float32(2392 / 4096 * 32)
    cc = np.float32(10000.0)

    div_m1 = np.float32(1.0) / m1
    div_m2 = np.float32(1.0) / m2

    numerator = np.maximum((x ** div_m2) - c1, 0)

    y = (numerator / (c2 - c3 * (x ** div_m2))) ** div_m1

    return y * cc


def st2084_inverse_eotf_fp32(x):
    ll = x / np.float32(10000.0)
    m1 = np.float32(2610 / 16384)
    m2 = np.float32(2523 / 4096 * 128)
    c1 = np.float32(3424 / 4096)
    c2 = np.float32(2413 / 4096 * 32)
    c3 = np.float32(2392 / 4096 * 32)

    y = ((c1 + c2 * (ll ** m1)) / (1 + c3 * (ll ** m1))) ** m2

    return y


def simulate_windows_signal_processing():
    rec2100_pq_10bit = create_test_rgb_data()
    rec2100_pq_fp32 = rec2100_pq_10bit / np.float32(1023.0)
    rec2100_linear = st2084_eotf_fp32(x=rec2100_pq_fp32)
    print(rec2100_linear)


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

    # simulate_windows_signal_processing()
    from colour.models import eotf_inverse_ST2084, eotf_ST2084
    rgb = np.array([[0, 0, 0], [1, 16, 18], [768, 769, 770], [511, 512, 1023]], dtype=np.uint16)
    y_fp64 = eotf_ST2084(rgb / 1023)
    # y_fp32 = st2084_eotf_fp32(rgb/np.float32(1023))
    y_fp32 = y_fp64.astype(np.float32)
    y_fp16 = y_fp64.astype(np.float16)

    # print(y_fp64 - y_fp32)
    # print(y_fp64 - y_fp16)

    st2084_fp64 = np.round(eotf_inverse_ST2084(y_fp64) * 1023).astype(np.uint16)
    st2084_fp16 = np.round(st2084_inverse_eotf_fp32(y_fp16) * 1023).astype(np.uint16)
    print(st2084_fp64)
    print(st2084_fp16)
