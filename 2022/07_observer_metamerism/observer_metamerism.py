# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
from difflib import diff_bytes
import os
from turtle import color
import requests

# import third-party libraries
import numpy as np
from colour.continuous import MultiSignals, Signal
from colour import sd_to_XYZ, MultiSpectralDistributions, MSDS_CMFS,\
    SDS_ILLUMINANTS, SpectralShape, SpectralDistribution
from colour.algebra import vector_dot
from colour.utilities import tstack
from colour.io import write_image

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import color_space as cs
import plot_utility as pu
from spectrum import DisplaySpectrum, calc_rgb_to_xyz_matrix_from_spectral_distribution, create_display_sd,\
    CIE1931_CMFS, CIE2012_CMFS, ILLUMINANT_E, START_WAVELENGTH,\
    STOP_WAVELENGTH, WAVELENGTH_STEP, trim_and_interpolate_in_advance,\
    calc_xyz_to_rgb_matrix_from_spectral_distribution, trim_and_iterpolate
import color_space as cc

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


DEFAULT_SPECTRAL_SHAPE = SpectralShape(
    START_WAVELENGTH, STOP_WAVELENGTH, WAVELENGTH_STEP)

SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS = SpectralShape(390, 730, 1)


def download_file(url, color_checker_sr_fname):
    response = requests.get(url)
    open(color_checker_sr_fname, "wb").write(response.content)


def prepaere_color_checker_sr_data():
    """
    Returns
    -------
    MultiSpectralDistributions
        multi-spectral distributions data.
    """
    color_checker_sr_fname = "./ref_data/color_checker_sr.txt"
    data = np.loadtxt(
        fname=color_checker_sr_fname, delimiter='\t', skiprows=1).T
    domain = np.arange(380, 740, 10)
    color_checker_signals = MultiSignals(data=data, domain=domain)
    color_checker_sds = MultiSpectralDistributions(data=color_checker_signals)

    return color_checker_sds


def color_checker_calc_sd_to_XYZ(
        spectral_shape=DEFAULT_SPECTRAL_SHAPE,
        cmfs=MSDS_CMFS['CIE 1931 2 Degree Standard Observer'],
        illuminant=SDS_ILLUMINANTS['D65']):
    color_checker_sds = prepaere_color_checker_sr_data()
    color_checker_sds = trim_and_iterpolate(color_checker_sds, spectral_shape)
    cmfs_trimed = trim_and_iterpolate(cmfs, spectral_shape)
    illuminant_intp = trim_and_iterpolate(illuminant, spectral_shape)
    # print(illuminant)
    XYZ = sd_to_XYZ(
        sd=color_checker_sds, cmfs=cmfs_trimed, illuminant=illuminant_intp)

    return XYZ


def create_color_checker_plus_d65_sd():
    color_checker_sr_fname = "./ref_data/color_checker_sr.txt"
    data = np.loadtxt(
        fname=color_checker_sr_fname, delimiter='\t', skiprows=1).T
    domain = np.arange(380, 740, 10)
    data_white = np.ones((len(domain), 1)) * 1.0
    data = np.append(data, data_white, axis=1)
    color_checker_signals = MultiSignals(data=data, domain=domain)
    color_checker_plut_d65_sds = MultiSpectralDistributions(
        data=color_checker_signals)

    return color_checker_plut_d65_sds


def calc_cc_plus_d65_xyz_for_each_cmfs(cmfs_list):
    """
    Parameters
    ----------
    cmfs_list : List
        A list of MultiSpectralDistributions

    Returns
    -------
    ndarray
        XYZ data.
        Shape is (num_of_cmfs, num_of_patch, 3).
    """
    spectral_shape = SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS
    color_checker_plut_d65_sds = create_color_checker_plus_d65_sd()
    color_checker_plut_d65_sds = trim_and_iterpolate(
        color_checker_plut_d65_sds, spectral_shape)
    illuminant = SDS_ILLUMINANTS['D65']
    illuminant = trim_and_iterpolate(illuminant, spectral_shape)

    # dummy_large_xyz = calc_cc_plus_d65_xyz(cmfs=cmfs_list[0])
    num_of_cmfs = len(cmfs_list)
    # num_of_patch = dummy_large_xyz.shape[0]
    num_of_patch = color_checker_plut_d65_sds.values.shape[1]

    large_xyz_out_buf = np.zeros((num_of_cmfs, num_of_patch, 3))
    for idx, cmfs in enumerate(cmfs_list):
        cmfs = trim_and_iterpolate(cmfs, spectral_shape)
        large_xyz = sd_to_XYZ(
            sd=color_checker_plut_d65_sds, cmfs=cmfs, illuminant=illuminant)
        large_xyz_out_buf[idx] = large_xyz

    return large_xyz_out_buf


def calc_xyz_to_rgb_matrix_each_display_sd_each_cmfs(
        display_sd_list, cmfs_list):
    """
    Returns
    -------
    ndarray
        Matrix list.
        Shape is (num_of_display, num_of_cmfs, 3, 3)
    """
    spectral_shape = SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS
    xyz_to_rgb_mtx_list = np.zeros(
        (len(display_sd_list), len(cmfs_list), 3, 3))
    for d_idx, display_sd in enumerate(display_sd_list):
        for c_idx, cmfs in enumerate(cmfs_list):
            mtx = calc_xyz_to_rgb_matrix_from_spectral_distribution(
                spd=display_sd, cmfs=cmfs, spectral_shape=spectral_shape)
            xyz_to_rgb_mtx_list[d_idx, c_idx] = mtx

    return xyz_to_rgb_mtx_list


def calc_tristimulus_value_for_each_sd_patch_cmfs(
        large_xyz, xyz_to_rgb_mtx):
    """
    Parameters
    ----------
    large_xyz : ndarray
        XYZ. shape is (num_of_cmfs, 25, 3)
    xyz_to_rgb_mtx : ndarray
        XYZ to RGB matrix.
        shape is (num_of_display, num_of_cmfs, 3, 3)

    Returns
    -------
    ndarray
        RGB value.
        Shape is (num_of_display, num_of_cmfs, num_of_patch, 3)
    """
    num_of_display = xyz_to_rgb_mtx.shape[0]
    num_of_cmfs = xyz_to_rgb_mtx.shape[1]
    num_of_patch = large_xyz.shape[1]
    out_rgb = np.zeros((num_of_display, num_of_cmfs, num_of_patch, 3))
    for d_idx in range(num_of_display):
        for c_idx in range(num_of_cmfs):
            mtx = xyz_to_rgb_mtx[d_idx, c_idx]
            large_xyz_temp = large_xyz[c_idx]
            rgb = vector_dot(mtx, large_xyz_temp)
            out_rgb[d_idx, c_idx] = rgb

    return out_rgb / 100


def create_modified_display_sd_based_on_rgb_gain_core(
        sd: MultiSpectralDistributions, rgb):
    """
    Parameters
    ----------
    sd : DisplaySpectrum
        display spectrum
    """
    r_sd = sd.values[..., 0]
    g_sd = sd.values[..., 1]
    b_sd = sd.values[..., 2]
    r_sd_new = r_sd * rgb[0]
    g_sd_new = g_sd * rgb[1]
    b_sd_new = b_sd * rgb[2]
    w_sd_new = r_sd_new + g_sd_new + b_sd_new
    domain = sd.domain
    gained_signal = Signal(data=w_sd_new, domain=domain)
    modified_sd = SpectralDistribution(data=gained_signal)

    return modified_sd


def create_modified_display_sd_based_on_rgb_gain(
        display_sd_list, rgb_list):
    """
    Parameters
    ----------
    display_sd_list : list
        A list of the display spectrum.
    rgb_list : ndarray
        RGB value for reproducting the specific color.
        Shape is (num_of_display, num_of_cmfs, num_of_patch)

    Returns
    -------
    A list of SpectralDistribution
        Shape is (num_of_display, num_of_cmfs, num_of_patch)
    """
    num_of_display = rgb_list.shape[0]
    num_of_cmfs = rgb_list.shape[1]
    num_of_patch = rgb_list.shape[2]
    out_buf = [
        [[0] * num_of_patch for ii in range(num_of_cmfs)]
        for jj in range(num_of_display)]
    spectral_shape = SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS

    for d_idx in range(num_of_display):
        sd = display_sd_list[d_idx]
        for c_idx in range(num_of_cmfs):
            for p_idx in range(num_of_patch):
                out_sd = create_modified_display_sd_based_on_rgb_gain_core(
                    sd=sd, rgb=rgb_list[d_idx, c_idx, p_idx])
                out_sd = trim_and_iterpolate(out_sd, spectral_shape)
                out_buf[d_idx][c_idx][p_idx] = out_sd

    return out_buf


def calc_XYZ_from_calibrated_display_sd_usin_cie1931(sd_list, rgb_list):
    """
    Parameters
    ----------
    sd_list : list
        A list of the display spectrum.
    rgb_list : ndarray
        RGB value for reproducting the specific color.
        Shape is (num_of_display, num_of_cmfs, num_of_patch)

    Returns
    -------
    ndarray
        A list of XYZ.
        Shape is (num_of_display, num_of_cmfs, num_of_patch, 3)
    """
    num_of_display = rgb_list.shape[0]
    num_of_cmfs = rgb_list.shape[1]
    num_of_patch = rgb_list.shape[2]
    out_buf = np.zeros((num_of_display, num_of_cmfs, num_of_patch, 3))

    spectral_shape = SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS
    cmfs = trim_and_iterpolate(CIE1931_CMFS, spectral_shape)
    illuminant = trim_and_iterpolate(ILLUMINANT_E, spectral_shape)

    for d_idx in range(num_of_display):
        for c_idx in range(num_of_cmfs):
            for p_idx in range(num_of_patch):
                sd_data = sd_list[d_idx][c_idx][p_idx]
                xyz = sd_to_XYZ(
                    sd=sd_data, cmfs=cmfs, illuminant=illuminant)
                out_buf[d_idx, c_idx, p_idx] = xyz / 100

    return out_buf


def calc_display_sd_using_metamerism(
        large_xyz, base_msd):
    """
    Parameters
    ----------
    large_xyz : ndarray
        Target XYZ value. Its shape must be (N, 3).
    base_msd : MultiSpectralDistributions
        Default display spectral distribution.

    Returns
    -------
    MultiSpectralDistributions
        Display spectral distribution. Its shape is (W, N).
        "W" means the number of wavelengths.
    """
    ds = DisplaySpectrum(msd=base_msd)
    xyz_to_rgb_mtx = ds.get_rgb_to_xyz_mtx()

    rgb_gain = vector_dot(xyz_to_rgb_mtx, large_xyz) / 100
    rgb_gain = np.clip(rgb_gain, 0.0, 1.0)

    metamerism_spectrum = ds.calc_msd_from_rgb_gain(rgb=rgb_gain)

    return metamerism_spectrum


def debug_numpy_mult_check():
    r_coef = np.arange(6) + 1
    r_coef = r_coef.reshape(1, 6)
    sd = np.linspace(0, 1, 8).reshape(8, 1)
    print(r_coef)
    print(sd)
    yy = r_coef * sd
    print(yy)


def calc_mismatch_large_xyz_using_two_cmfs(
        msd=None, cmfs2=CIE2012_CMFS):
    """
    Returns
    -------
    large_xyz_cmfs2 : ndarray
        A XYZ value calculated from spectral reflectance using cmfs2.
    mismatch_large_xyz : ndarray
        A XYZ value calculated from display spectral distribution
        using cmfs2.
        display spectral distribution is based on cmfs1.
    """
    cmfs1 = CIE1931_CMFS
    spectral_shape = DEFAULT_SPECTRAL_SHAPE
    illuminant = ILLUMINANT_E
    cmfs1 = trim_and_iterpolate(cmfs1, spectral_shape)
    cmfs2 = trim_and_iterpolate(cmfs2, spectral_shape)
    illuminant = trim_and_iterpolate(illuminant, spectral_shape)

    cc_large_xyz_1931 = color_checker_calc_sd_to_XYZ(
        spectral_shape=spectral_shape, cmfs=cmfs1)
    large_xyz_cmfs2 = color_checker_calc_sd_to_XYZ(
        spectral_shape=spectral_shape, cmfs=cmfs2)

    cc_spectrum = calc_display_sd_using_metamerism(
        large_xyz=cc_large_xyz_1931, base_msd=msd)
    cc_spectrum = trim_and_iterpolate(cc_spectrum, spectral_shape)

    mismatch_large_xyz = sd_to_XYZ(
        sd=cc_spectrum, cmfs=cmfs2, illuminant=illuminant)

    return large_xyz_cmfs2, mismatch_large_xyz


def draw_mismatch_cmfs2_color_checker_image(
        large_xyz, mismatch_large_xyz, fname="./figure/hoge.png"):
    rgb = cs.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=cs.BT709)
    rgb = np.clip(rgb / 100, 0.0, 1.0)

    rgb_mismatch = cs.large_xyz_to_rgb(
        xyz=mismatch_large_xyz, color_space_name=cs.BT709)
    rgb_mismatch = np.clip(rgb_mismatch / 100, 0.0, 1.0)

    cc_image = tpg.plot_color_checker_image(
        rgb=rgb, rgb2=rgb_mismatch, size=(1280, 720))

    cc_image_srgb = tf.oetf(cc_image, tf.SRGB)

    tpg.img_wirte_float_as_16bit_int(fname, cc_image_srgb)


def debug_calc_msd_from_rgb_gain(
        r_mu=620, r_sigma=12, g_mu=535, g_sigma=18, b_mu=458, b_sigma=8,
        fname="./figure/sample222.png"):
    msd = create_display_sd(
        r_mu=r_mu, r_sigma=r_sigma,
        g_mu=g_mu, g_sigma=g_sigma,
        b_mu=b_mu, b_sigma=b_sigma,
        normalize_y=True)
    spectral_shape = DEFAULT_SPECTRAL_SHAPE
    cmfs_1931 = CIE1931_CMFS
    cmfs_2012 = CIE2012_CMFS
    illuminant = ILLUMINANT_E
    cmfs_1931 = trim_and_iterpolate(cmfs_1931, spectral_shape)
    cmfs_2012 = trim_and_iterpolate(cmfs_2012, spectral_shape)
    illuminant = trim_and_iterpolate(illuminant, spectral_shape)

    cc_large_xyz = color_checker_calc_sd_to_XYZ(
        spectral_shape=spectral_shape, cmfs=cmfs_1931)
    cc_large_xyz_2012 = color_checker_calc_sd_to_XYZ(
        spectral_shape=spectral_shape, cmfs=cmfs_2012)

    cc_spectrum = calc_display_sd_using_metamerism(
        large_xyz=cc_large_xyz, base_msd=msd)
    cc_spectrum = trim_and_iterpolate(cc_spectrum, spectral_shape)

    xyz_1931 = sd_to_XYZ(
        sd=cc_spectrum, cmfs=cmfs_1931, illuminant=illuminant)
    xyz_2012 = sd_to_XYZ(
        sd=cc_spectrum, cmfs=cmfs_2012, illuminant=illuminant)

    rgb_1931 = cs.large_xyz_to_rgb(
        xyz=xyz_1931, color_space_name=cs.BT709)
    rgb_2012 = cs.large_xyz_to_rgb(
        xyz=xyz_2012, color_space_name=cs.BT709)

    rgb_1931 = np.clip(rgb_1931 / 100, 0.0, 1.0)
    rgb_2012 = np.clip(rgb_2012 / 100, 0.0, 1.0)

    img_cat = tpg.plot_color_checker_image(
        rgb=rgb_1931, rgb2=rgb_2012, size=(1280, 720))
    img_1931 = tpg.plot_color_checker_image(
        rgb=rgb_1931, rgb2=None, size=(1280, 720))
    img_2012 = tpg.plot_color_checker_image(
        rgb=rgb_2012, rgb2=None, size=(1280, 720))

    tpg.img_wirte_float_as_16bit_int(fname, img_cat**(1/2.4))
    tpg.img_wirte_float_as_16bit_int(fname+"_1931.png", img_1931**(1/2.4))
    tpg.img_wirte_float_as_16bit_int(fname+"_2012.png", img_2012**(1/2.4))

    # print(rgb_gain)


def calc_delta_xyz(xyz1, xyz2):
    diff_x = xyz1[..., 0] - xyz2[..., 0]
    diff_y = xyz1[..., 1] - xyz2[..., 1]
    diff_z = xyz1[..., 2] - xyz2[..., 2]

    delta_xyz = ((diff_x ** 2) + (diff_y ** 2) + (diff_z ** 2)) ** 0.5

    return delta_xyz


def debug_plot_color_checker_delta_xyz(
        ok_xyz, ng_xyz_709, ng_xyz_p3, ng_xyz_2020, fname_suffix="0"):
    delta_709 = calc_delta_xyz(ok_xyz, ng_xyz_709)
    delta_p3 = calc_delta_xyz(ok_xyz, ng_xyz_p3)
    delta_2020 = calc_delta_xyz(ok_xyz, ng_xyz_2020)

    # label = [
    #     "dark skin", "light skin", "blue sky", "foliage", "blue flower",
    #     "bluish green", "orange", "purplish blue", "moderate red", "purple",
    #     "yellow green", "orange yellow", "blue", "green", "red", "yellow",
    #     "magenta", "cyan", "white 9.5", "neutral 8", "neutral 6.5",
    #     "neutral 5", "neutral 3.5", "black 2"]
    label = [
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"]

    x = np.arange(24) + 1
    x_709 = x - 0.25
    x_p3 = x + 0.0
    x_2020 = x + 0.25

    fig, ax1 = pu.plot_1_graph()
    ax1.bar(
        x_709, delta_709, width=0.25, color=pu.RED,
        align="center", label="BT.709")
    ax1.bar(
        x_p3, delta_p3, width=0.25, color=pu.GREEN,
        align="center", label="DCI-P3")
    ax1.bar(
        x_2020, delta_2020, width=0.25, color=pu.BLUE,
        align="center", label="BT.2020")

    ax1.set_xticks(x, label)
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname=f"./figure/delta_xyz_{fname_suffix}.png")


def load_2deg_151_cmfs():
    """
    source: https://www.rit.edu/science/sites/rit.edu.science/files/2019-01/MCSL-Observer_Function_Database.pdf
    """
    fname_x = "./ref_data/RIT_MCSL_CMFs_151_02deg_x.csv"
    fname_y = "./ref_data/RIT_MCSL_CMFs_151_02deg_y.csv"
    fname_z = "./ref_data/RIT_MCSL_CMFs_151_02deg_z.csv"

    xx_base = np.loadtxt(fname=fname_x, delimiter=',')
    domain = xx_base[..., 0]
    xx = xx_base[..., 1:]
    yy = np.loadtxt(fname=fname_y, delimiter=',')[..., 1:]
    zz = np.loadtxt(fname=fname_z, delimiter=',')[..., 1:]

    num_of_cmfs = xx.shape[1]

    cmfs_array_151 = []

    for cmfs_idx in range(num_of_cmfs):
        sd = tstack([xx[..., cmfs_idx], yy[..., cmfs_idx], zz[..., cmfs_idx]])
        signals = MultiSignals(data=sd, domain=domain)
        sds = MultiSpectralDistributions(data=signals)
        cmfs_array_151.append(sds)

    return cmfs_array_151


def load_2deg_10_cmfs():
    """
    source: https://www.rit.edu/science/sites/rit.edu.science/files/2019-01/MCSL-Observer_Function_Database.pdf
    """
    fname_x = "./ref_data/RIT_MSCL_CMFs_10_02deg_x.csv"
    fname_y = "./ref_data/RIT_MSCL_CMFs_10_02deg_y.csv"
    fname_z = "./ref_data/RIT_MSCL_CMFs_10_02deg_z.csv"

    xx_base = np.loadtxt(fname=fname_x, delimiter=',')
    domain = xx_base[..., 0]
    xx = xx_base[..., 1:]
    yy = np.loadtxt(fname=fname_y, delimiter=',')[..., 1:]
    zz = np.loadtxt(fname=fname_z, delimiter=',')[..., 1:]

    num_of_cmfs = xx.shape[1]

    cmfs_array_10 = []

    for cmfs_idx in range(num_of_cmfs):
        sd = tstack([xx[..., cmfs_idx], yy[..., cmfs_idx], zz[..., cmfs_idx]])
        signals = MultiSignals(data=sd, domain=domain)
        sds = MultiSpectralDistributions(data=signals)
        cmfs_array_10.append(sds)

    return cmfs_array_10


def debug_plot_151_cmfs():
    cmfs_array = load_2deg_151_cmfs()

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="151 color-normal human cmfs",
        graph_title_size=None,
        xlabel="Wavelength [nm]",
        ylabel="Tristimulus Values",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=1,
        minor_xtick_num=None,
        minor_ytick_num=None)
    for cmfs in cmfs_array:
        ax1.plot(
            cmfs.wavelengths, cmfs.values[..., 0], '-',
            color=pu.RED, alpha=1/5)
        ax1.plot(
            cmfs.wavelengths, cmfs.values[..., 1], '-',
            color=pu.GREEN, alpha=1/5)
        ax1.plot(
            cmfs.wavelengths, cmfs.values[..., 2], '-',
            color=pu.BLUE, alpha=1/5)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname="./figure/cmfs_151.png")


def debug_plot_10_cmfs():
    cmfs_array = load_2deg_10_cmfs()
    spectral_shape = SpectralShape(300, 830, 1)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="10 categorical observers (xyz 2 degree)",
        graph_title_size=None,
        xlabel="Wavelength [nm]",
        ylabel="Tristimulus Values",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=1,
        minor_xtick_num=None,
        minor_ytick_num=None)
    for cmfs in cmfs_array:
        cmfs = cmfs.extrapolate(spectral_shape)
        cmfs = cmfs.interpolate(spectral_shape)
        ax1.plot(
            cmfs.wavelengths, cmfs.values[..., 0], '-',
            color=pu.RED, alpha=1/2)
        ax1.plot(
            cmfs.wavelengths, cmfs.values[..., 1], '-',
            color=pu.GREEN, alpha=1/2)
        ax1.plot(
            cmfs.wavelengths, cmfs.values[..., 2], '-',
            color=pu.BLUE, alpha=1/2)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname="./figure/cmfs_10.png")


def create_709_p3_2020_display_sd():
    bt709_msd = create_display_sd(
        r_mu=649, r_sigma=35, g_mu=539, g_sigma=33, b_mu=460, b_sigma=13,
        normalize_y=True)
    p3_msd = create_display_sd(
        r_mu=620, r_sigma=12, g_mu=535, g_sigma=18, b_mu=458, b_sigma=8,
        normalize_y=True)
    bt2020_msd = create_display_sd(
        r_mu=639, r_sigma=3, g_mu=530, g_sigma=4, b_mu=465, b_sigma=4,
        normalize_y=True)

    return bt709_msd, p3_msd, bt2020_msd


def debug_calc_and_plot_metamerism_delta():
    bt709_msd, p3_msd, bt2020_msd = create_709_p3_2020_display_sd()
    cmfs_list = load_2deg_10_cmfs()
    result_709_list = []
    result_p3_list = []
    result_2020_list = []
    ref_xyz_list = []

    for idx, cmfs in enumerate(cmfs_list):
        print(f"calc cmfs idx: {idx}")
        ok_xyz, ng_xyz_709 =\
            calc_mismatch_large_xyz_using_two_cmfs(
                msd=bt709_msd, cmfs2=cmfs)
        ok_xyz, ng_xyz_p3 =\
            calc_mismatch_large_xyz_using_two_cmfs(
                msd=p3_msd, cmfs2=cmfs)
        ok_xyz, ng_xyz_2020 =\
            calc_mismatch_large_xyz_using_two_cmfs(
                msd=bt2020_msd, cmfs2=cmfs)
        ref_xyz_list.append(ok_xyz)
        result_709_list.append(ng_xyz_709)
        result_p3_list.append(ng_xyz_p3)
        result_2020_list.append(ng_xyz_2020)

    for cmfs_idx in range(len(cmfs_list)):
        debug_plot_color_checker_delta_xyz(
            ok_xyz=ref_xyz_list[cmfs_idx],
            ng_xyz_709=result_709_list[cmfs_idx],
            ng_xyz_p3=result_p3_list[cmfs_idx],
            ng_xyz_2020=result_2020_list[cmfs_idx],
            fname_suffix=f"cmfs_idx-{cmfs_idx:02d}")


def debug_save_white_patch(large_xyz):
    # org_shape = large_xyz.shape
    # large_xyz = large_xyz.reshape(1, -1, 3)
    rgb = cc.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=cs.BT709,
        xyz_white=cs.D65, rgb_white=cs.D65)
    # rgb = rgb.reshape(org_shape)
    print(f"max={np.max(rgb[:, :, 24])}, min={np.min(rgb[:, :, 24])}")
    rgb = rgb / np.max(rgb[:, :, 24])
    img_size = 200
    base_img = np.ones((img_size, img_size, 3))
    v_img_buf = []
    for d_idx in range(3):
        h_img_buf = []
        for c_idx in range(10):
            img = base_img * rgb[d_idx, c_idx, 24]  # 24 is white
            tpg.draw_outline(img, fg_color=[0.2, 0.2, 0.2], outline_width=1)
            h_img_buf.append(img)
        v_img_buf.append(np.hstack(h_img_buf))
    out_img = np.vstack(v_img_buf)
    out_img = tf.oetf(np.clip(out_img, 0.0, 1.0), tf.SRGB)

    write_image(out_img, "./debug/white_with_multi_cmfs.png")


def debug_verify_calibrated_sd(modified_sd_list, cmfs_list, large_xyz):
    spectral_shape = SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS
    illuminant = trim_and_iterpolate(ILLUMINANT_E, spectral_shape)
    for d_idx in range(3):
        for c_idx in range(10):
            cmfs = cmfs_list[c_idx]
            cmfs = trim_and_iterpolate(cmfs, spectral_shape)
            for p_idx in range(25):
                if p_idx < 24:
                    continue
                sd = modified_sd_list[d_idx][c_idx][p_idx]
                sd = trim_and_iterpolate(sd, spectral_shape)
                xyz = sd_to_XYZ(
                    sd=sd, cmfs=cmfs, illuminant=illuminant)
                ref_xyz = large_xyz[c_idx, p_idx]
                diff = ref_xyz - xyz
                msg = f"(d, c, p)=({d_idx}, {c_idx}, {p_idx}), "
                msg += f"{ref_xyz}-{xyz}={diff}"
                print(msg)


def debug_func():
    # debug_numpy_mult_check()
    bt709_msd = create_display_sd(
        r_mu=649, r_sigma=35, g_mu=539, g_sigma=33, b_mu=460, b_sigma=13,
        normalize_y=True)
    p3_msd = create_display_sd(
        r_mu=620, r_sigma=12, g_mu=535, g_sigma=18, b_mu=458, b_sigma=8,
        normalize_y=True)
    bt2020_msd = create_display_sd(
        r_mu=639, r_sigma=3, g_mu=530, g_sigma=4, b_mu=465, b_sigma=4,
        normalize_y=True)

    # ok_xyz, ng_xyz_709 =\
    #     calc_mismatch_large_xyz_using_two_cmfs(
    #         msd=bt709_msd, cmfs2=CIE2012_CMFS)
    # draw_mismatch_cmfs2_color_checker_image(
    #     large_xyz=ok_xyz, mismatch_large_xyz=ng_xyz_709,
    #     fname="./figure/bt709_2012_cc.png")

    # ok_xyz, ng_xyz_p3 =\
    #     calc_mismatch_large_xyz_using_two_cmfs(
    #         msd=p3_msd, cmfs2=CIE2012_CMFS)
    # draw_mismatch_cmfs2_color_checker_image(
    #     large_xyz=ok_xyz, mismatch_large_xyz=ng_xyz_p3,
    #     fname="./figure/p3_2012_cc.png")

    # ok_xyz, ng_xyz_2020 =\
    #     calc_mismatch_large_xyz_using_two_cmfs(
    #         msd=bt2020_msd, cmfs2=CIE2012_CMFS)
    # draw_mismatch_cmfs2_color_checker_image(
    #     large_xyz=ok_xyz, mismatch_large_xyz=ng_xyz_2020,
    #     fname="./figure/bt2020_2012_cc.png")

    # debug_plot_color_checker_delta_xyz(
    #     ok_xyz=ok_xyz, ng_xyz_709=ng_xyz_709,
    #     ng_xyz_p3=ng_xyz_p3, ng_xyz_2020=ng_xyz_2020)

    # debug_plot_151_cmfs()
    # debug_plot_10_cmfs()

    # debug_calc_and_plot_metamerism_delta()

    # inter-observer simulation
    cmfs_list = load_2deg_10_cmfs()
    large_xyz_il_d65 = calc_cc_plus_d65_xyz_for_each_cmfs(cmfs_list=cmfs_list)
    display_sd_list = [bt709_msd, p3_msd, bt2020_msd]
    xyz_to_rgb_mtx = calc_xyz_to_rgb_matrix_each_display_sd_each_cmfs(
        display_sd_list=display_sd_list, cmfs_list=cmfs_list)
    rgb = calc_tristimulus_value_for_each_sd_patch_cmfs(
        large_xyz=large_xyz_il_d65, xyz_to_rgb_mtx=xyz_to_rgb_mtx)
    # rgb = rgb / np.max(rgb)
    modified_sd_list = create_modified_display_sd_based_on_rgb_gain(
        display_sd_list=display_sd_list, rgb_list=rgb)
    large_xyz_1931 = calc_XYZ_from_calibrated_display_sd_usin_cie1931(
        sd_list=modified_sd_list, rgb_list=rgb)
    # print(large_xyz)

    np.save("./debug/calibrated_xyz.npy", large_xyz_1931)
    large_xyz_1931 = np.load("./debug/calibrated_xyz.npy")
    debug_save_white_patch(large_xyz=large_xyz_1931)
    debug_verify_calibrated_sd(
        modified_sd_list=modified_sd_list, cmfs_list=cmfs_list,
        large_xyz=large_xyz_il_d65)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
