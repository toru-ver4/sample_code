# -*- coding: utf-8 -*-
"""
ACES の Output Transfrom について少し調べる
===============================================

Description.

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from colour import write_image, read_image

# import my libraries
import plot_utility as pu
import aces_rrt_odt as aro
import test_pattern_generator2 as tpg
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

RAMP_FILE_NAME = "./img/white_ramp.exr"
HDR_1K_LINEAR_NAME = "./img/hdr_home_1k.exr"

SAMPLE_NUM = 1024
REF_VAL = 1.0
MIN_EXPOSURE = -5
MAX_EXPOSURE = 4

CTL_MODULE_PATH = "/work/src/2020/015_research_aces_ot/ctl/lib"
RRT_PATH = "./ctl/rrt/RRT.ctl"
SDR_CINEMA_ODT_PATH = "./ctl/odt/dcdm/ODT.Academy.DCDM.ctl"
SDR_ODT_PATH = "./ctl/odt/rec709/ODT.Academy.Rec709_100nits_dim.ctl"
EDR_CINEMA_OT_PATH = "./ctl/outputTransforms/RRTODT.Academy.P3D65_108nits_7.2nits_ST2084.ctl"
HDR_HOME_1K_PATH = "./ctl/outputTransforms/RRTODT.Academy.Rec2020_1000nits_15nits_ST2084.ctl"
HDR_HOME_4K_PATH = "./ctl/outputTransforms/RRTODT.Academy.Rec2020_4000nits_15nits_ST2084.ctl"
INV_HDR_HOME_1K_PATH = "./ctl/outputTransforms/InvRRTODT.Academy.Rec2020_1000nits_15nits_ST2084.ctl"
INV_HDR_HOME_4K_PATH = "./ctl/outputTransforms/InvRRTODT.Academy.Rec2020_4000nits_15nits_ST2084.ctl"

YELLOW = np.array((255, 241, 0)) / 255
GREEN = np.array((3, 175, 122)) / 255
BLACK = np.array((0, 0, 0)) / 255
RED = np.array((255, 75, 0)) / 255
BLUE = np.array((0, 90, 255)) / 255


def make_and_save_baseic_ramp():
    x = tpg.get_log10_x_scale(SAMPLE_NUM, REF_VAL, MIN_EXPOSURE, MAX_EXPOSURE)
    img = np.dstack((x, x, x))
    write_image(img, RAMP_FILE_NAME, bit_depth='float32')
    return x


def log_scale_settings(ax1):
    """
    https://stackoverflow.com/questions/44078409/matplotlib-semi-log-plot-minor-tick-marks-are-gone-when-range-is-large
    """
    # Log Scale
    ax1.set_xscale('log', basex=10)
    ax1.set_yscale('log', basey=10)
    ax1.tick_params(
        which='major', direction='in', top=True, right=True,
        length=8)
    ax1.tick_params(
        which='minor', direction='in', top=True, right=True,
        length=5)
    major_locator = ticker.LogLocator(base=10, numticks=16)
    minor_locator = ticker.LogLocator(
        base=10, subs=[x * 0.1 for x in range(10)], numticks=16)
    ax1.get_xaxis().set_major_locator(major_locator)
    ax1.get_xaxis().set_minor_locator(minor_locator)
    ax1.get_xaxis().set_major_locator(ticker.LogLocator())
    ax1.grid(which='both', linestyle='-', alpha=0.4)
    ax1.patch.set_facecolor("#E0E0E0")


def plot_sdr_cinema(x):
    ctlrender_out_name = aro.apply_ctl_to_exr_image(
        img_list=[RAMP_FILE_NAME],
        ctl_list=[RRT_PATH, SDR_CINEMA_ODT_PATH],
        ctl_module_path=CTL_MODULE_PATH)
    after_img = read_image(ctlrender_out_name[0])
    luminance = (after_img[..., 1] ** 2.6) * 48
    # print(after_img.shape)
    # print(after_img.shape)

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="ACES v1.1 RRT+ODT",
        graph_title_size=None,
        xlabel="ACES Scene Reflectance Value",
        ylabel="Output Luminance (cd /m2)",
        axis_label_size=None,
        legend_size=17,
        linewidth=3,
        xlim=(0.00006, 4000),
        ylim=(0.00006, 4000))
    log_scale_settings(ax1)

    ax1.plot(x, luminance, c=YELLOW, label='SDR Cinmea 48nits')
    plt.legend(loc='lower right')
    plt.show()


def calc_rrt_and_odt(
        img_list=[RAMP_FILE_NAME], ctl_list=[EDR_CINEMA_OT_PATH], ext='.tiff'):
    ctlrender_out_name = aro.apply_ctl_to_exr_image(
        img_list=img_list,
        ctl_list=ctl_list,
        ctl_module_path=CTL_MODULE_PATH,
        out_ext=ext)
    after_img = read_image(ctlrender_out_name[0])
    return after_img, ctlrender_out_name[0]


def plot_rrt_odt(x, d):

    ax1 = pu.plot_1_graph(
        fontsize=25,
        figsize=(16, 12),
        graph_title="ACES v1.1 RRT+ODT",
        graph_title_size=None,
        xlabel="ACES Scene Reflectance Value",
        ylabel="Output Luminance (cd /m2)",
        axis_label_size=None,
        legend_size=30,
        linewidth=3,
        xlim=(0.00006, 4000),
        ylim=(0.0001, 5000))
    log_scale_settings(ax1)

    ax1.plot(x, d['sdr_cinema'], c=YELLOW, label='SDR Cinmea 48nits')
    ax1.plot(x, d['edr_cinema'], '-.', c=GREEN, label='EDR Cinmea 108nits')
    ax1.plot(x, d['sdr_home'], '--', c=BLACK, label='SDR Home 100nits')
    ax1.plot(x, d['hdr_home_1k'], '-', c=BLUE, label='HDR Home 1000nits')
    ax1.plot(x, d['hdr_home_4k'], '--', c=RED, label='HDR Home 4000nits')
    plt.legend(loc='lower right')
    plt.savefig("./figures/rrt_odt.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_rrt_odt_control():
    x = make_and_save_baseic_ramp()
    # SDR Cinema
    sdr_cinema, _ = calc_rrt_and_odt(
        img_list=[RAMP_FILE_NAME], ctl_list=[RRT_PATH, SDR_CINEMA_ODT_PATH])
    sdr_cinema = (sdr_cinema ** 2.6) * 48
    # EDR Cinema
    edr_cinema, _ = calc_rrt_and_odt(
        img_list=[RAMP_FILE_NAME], ctl_list=[EDR_CINEMA_OT_PATH])
    edr_cinema = tf.eotf_to_luminance(edr_cinema, tf.ST2084)
    # SDR Home
    sdr_home, _ = calc_rrt_and_odt(
        img_list=[RAMP_FILE_NAME], ctl_list=[RRT_PATH, SDR_ODT_PATH])
    sdr_home = tf.eotf_to_luminance(sdr_home, tf.GAMMA24)
    # HDR Home 1000nits
    hdr_home_1k, _ = calc_rrt_and_odt(
        img_list=[RAMP_FILE_NAME], ctl_list=[HDR_HOME_1K_PATH])
    hdr_home_1k = tf.eotf_to_luminance(hdr_home_1k, tf.ST2084)
    # HDR Home 4000nits
    hdr_home_4k, _ = calc_rrt_and_odt(
        img_list=[RAMP_FILE_NAME], ctl_list=[HDR_HOME_4K_PATH])
    hdr_home_4k = tf.eotf_to_luminance(hdr_home_4k, tf.ST2084)

    d = dict(
        sdr_cinema=sdr_cinema[..., 1], edr_cinema=edr_cinema[..., 1],
        sdr_home=sdr_home[..., 1], hdr_home_1k=hdr_home_1k[..., 1],
        hdr_home_4k=hdr_home_4k[..., 1])
    plot_rrt_odt(x, d)


def plot_edr_cinema_aces_mapping(
        x, edr_cinema_from_inv, x2, edr_cinema_direct_ot):

    ax1 = pu.plot_1_graph(
        fontsize=25,
        figsize=(16, 12),
        graph_title="ACES v1.1 RRT+ODT",
        graph_title_size=None,
        xlabel="Pixel Luminance in HDR Home (cd/m2)",
        ylabel="Pixel Luminance in EDR Cinema (cd/m2)",
        axis_label_size=None,
        legend_size=26,
        linewidth=3,
        xlim=(0.001, 4000),
        ylim=(0.001, 400))
    log_scale_settings(ax1)

    ax1.plot(
        x, edr_cinema_from_inv, '-', c=GREEN,
        label='EDR Cinmea ACES Mapping (generated from HDR Home 4000nits)')
    # ax1.plot(x2 * 100, edr_cinema_direct_ot, '--', c=BLACK,
    #          label="EDR Cinema 108nits")
    plt.legend(loc='upper left')
    plt.savefig("./figures/HDR_Cinema_ACES_mapping.png",
                bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_edr_cinema_aces_mapping_control():
    x = make_and_save_baseic_ramp()

    # HDR Home 1000nits
    hdr_home_4k, after_ot_name = calc_rrt_and_odt(
        img_list=[RAMP_FILE_NAME], ctl_list=[HDR_HOME_4K_PATH])
    hdr_home_4k_linar = tf.eotf_to_luminance(hdr_home_4k[..., 0], tf.ST2084)

    # inv ot
    aces_linear, after_inv_ot_name = calc_rrt_and_odt(
        img_list=[after_ot_name], ctl_list=[INV_HDR_HOME_4K_PATH],
        ext='.exr')

    edr_cinema, _ = calc_rrt_and_odt(
        img_list=[after_inv_ot_name], ctl_list=[EDR_CINEMA_OT_PATH])
    edr_cinema_luminance = tf.eotf_to_luminance(edr_cinema[..., 1], tf.ST2084)

    # EDR Cinema 108nits
    edr_cinema_direct_ot, _ = calc_rrt_and_odt(
        img_list=[RAMP_FILE_NAME], ctl_list=[EDR_CINEMA_OT_PATH])
    edr_cinema_direct_ot_luminance = tf.eotf_to_luminance(
        edr_cinema_direct_ot[..., 1], tf.ST2084)

    plot_edr_cinema_aces_mapping(
        x=hdr_home_4k_linar, edr_cinema_from_inv=edr_cinema_luminance,
        x2=x, edr_cinema_direct_ot=edr_cinema_direct_ot_luminance)


def main_func():
    plot_rrt_odt_control()
    # plot_edr_cinema_aces_mapping_control()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # x = make_and_save_baseic_ramp()
    # plot_sdr_cinema(x)
    main_func()
