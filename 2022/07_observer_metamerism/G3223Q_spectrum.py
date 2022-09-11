# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import SpectralShape, XYZ_to_xy, xy_to_XYZ, SDS_ILLUMINANTS
from colour.utilities import tstack

# import my libraries
from spectrum import START_WAVELENGTH, STOP_WAVELENGTH, WAVELENGTH_STEP,\
        MultiSignals, MultiSpectralDistributions,\
        trim_and_iterpolate, ILLUMINANT_E, CIE1931_CMFS, sd_to_XYZ
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


SPECTRAL_SHAPE = SpectralShape(
    START_WAVELENGTH, STOP_WAVELENGTH, WAVELENGTH_STEP)
ILLUMINANT_D65 = SDS_ILLUMINANTS['D65']


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


def plot_spectrum():
    data_name = "./ref_data/G3223Q_Adjusted_default.csv"
    data = np.loadtxt(data_name, delimiter=',')

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Title",
        graph_title_size=None,
        xlabel="Wavelength [nm]",
        ylabel="Relative power",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[350, 400, 450, 500, 550, 600, 650, 700, 750],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(data[..., 0], data[..., 2] / np.max(data[..., 2]))
    pu.show_and_save(
        fig=fig, legend_loc=None, save_fname="./debug/G3223Q.png")


def calc_xyY():
    spectral_shape = SPECTRAL_SHAPE
    data_name = "./ref_data/G3223Q_Adjusted_default.csv"
    data = np.loadtxt(data_name, delimiter=',')
    signals = MultiSignals(
        domain=data[..., 0], data=data[..., 1:]/0xFFFF)
    msd = MultiSpectralDistributions(data=signals)
    illuminant_E = ILLUMINANT_E
    illuminant_D65 = ILLUMINANT_D65
    cmfs_list = load_2deg_10_cmfs()
    cmfs_list.append(CIE1931_CMFS)

    spd2 = trim_and_iterpolate(msd, spectral_shape)
    illuminant_E2 = trim_and_iterpolate(illuminant_E, spectral_shape)
    illuminant_D65_2 = trim_and_iterpolate(illuminant_D65, spectral_shape)

    out_buf = np.zeros((11, 2))

    for idx, cmfs in enumerate(cmfs_list):
        # calc display white point
        cmfs2 = trim_and_iterpolate(cmfs, spectral_shape)
        display_large_xyz = sd_to_XYZ(
            sd=spd2, cmfs=cmfs2, illuminant=illuminant_E2)
        display_xy = XYZ_to_xy(display_large_xyz)
        # print(display_xy)

        # calc cmfs's white point
        cmfs_white_large_xyz = sd_to_XYZ(
            sd=ILLUMINANT_E, cmfs=cmfs2, illuminant=illuminant_D65_2)
        cmfs_white_xy = XYZ_to_xy(cmfs_white_large_xyz)
        cmfs_white_xy = np.array(
            [[cmfs_white_xy[0], cmfs_white_xy[1]],
             [cmfs_white_xy[0], cmfs_white_xy[1]]])
        # print(cmfs_white_xy)
        diff_xy = display_xy - cmfs_white_xy
        delta_xy = ((diff_xy[..., 0] ** 2) + (diff_xy[..., 1] ** 2)) ** 0.5
        out_buf[idx] = delta_xy

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel="CMFs Index",
        ylabel="delta xy",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x for x in range(1, 12)],
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    x1 = np.arange(1, 12) - 0.2
    x2 = np.arange(1, 12) + 0.2
    bar_width = 0.4
    ax1.bar(x1, out_buf[..., 1], width=bar_width, label='factory preset')
    ax1.bar(x2, out_buf[..., 0], width=bar_width, label='manual adjustment')
    pu.show_and_save(
        fig=fig, legend_loc='upper right',
        save_fname="./debug/G3223Q_delta_xy.png")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    calc_xyY()
    # plot_spectrum()
