# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.continuous import MultiSignals
from colour import MultiSpectralDistributions, SpectralShape, MSDS_CMFS,\
    SDS_ILLUMINANTS, sd_to_XYZ, XYZ_to_xyY
from colour.io import write_image
from colour.utilities import tstack

# import my libraries
import font_control2 as fc2
import plot_utility as pu
import test_pattern_generator2 as tpg
import color_space as cs


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


CIE1931_NAME = "cie_2_1931"
CIE1931_CMFS = MultiSpectralDistributions(MSDS_CMFS[CIE1931_NAME])

ILLUMINANT_E = SDS_ILLUMINANTS['E']


def add_text_upper_left(img, text):
    text_draw_ctrl = fc2.TextDrawControl(
        text=text, font_color=[0.2, 0.2, 0.2],
        font_size=50, font_path=fc2.NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=5, stroke_fill=[0.0, 0.0, 0.0])

    # calc position
    _, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = 0
    pos_v = (text_height // 2)
    pos = (pos_h, pos_v)

    text_draw_ctrl.draw(img=img, pos=pos)


def create_measure_patch():
    width = 1920
    height = 1080
    sample_num = 5
    color_mask_list = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    color_info_list = ["R", "G", "B", "W"]
    val_list = np.linspace(0, 1, sample_num)
    for color_idx in range(len(color_mask_list)):
        base_img = np.ones((height, width, 3))
        for val in val_list:
            img = base_img * val * color_mask_list[color_idx]
            cv_8bit = np.array([val, val, val]) * color_mask_list[color_idx]
            cv_8bit = np.uint8(np.round((cv_8bit ** (1/2.2)) * 0xFF))
            text = f" (R, G, B) = ({cv_8bit[0]}, {cv_8bit[1]}, {cv_8bit[2]})"
            add_text_upper_left(img=img, text=text)

            color_name = color_info_list[color_idx]
            fname = f"./img_measure_patch/display_measure_patch_{color_name}_"
            fname += f"{np.max(cv_8bit):03d}.png"
            print(fname)
            write_image(img**(1/2.2), fname, 'uint8')


def load_display_spd(fname="./spd_measure_data/rgbw_spd.csv"):
    data = np.loadtxt(fname=fname, delimiter=",")
    data[..., 1:] = data[..., 1:] / 0xFFFF

    return data


def plot_each_color(
        ax, wl, data, color, color_name, val_list,
        linestyle='-', linewidth=3):
    for idx in range(4):
        label = f"{color_name}_{val_list[idx]:0.2f}"
        ax.plot(
            wl, data[..., idx], linestyle, color=color, alpha=1-0.15*(4-idx),
            label=label, lw=linewidth)


def plot_b_display_spectral_distribution():
    spd = load_display_spd(fname="./spd_measure_data/rgbw_spd.csv")
    wl = spd[..., 0]
    val_list = np.linspace(0, 1, 5)[1:]
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Spectral distribution",
        graph_title_size=None,
        xlabel="Wavelength [nm]",
        ylabel="Relative power",
        axis_label_size=None,
        legend_size=12,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=5,
        minor_ytick_num=4)
    plot_each_color(
        ax=ax1, wl=wl, data=spd[..., 2:6], color_name="B",
        color=pu.BLUE, val_list=val_list)
    plot_each_color(
        ax=ax1, wl=wl, data=spd[..., 7:11], color_name="G",
        color=pu.GREEN, val_list=val_list)
    plot_each_color(
        ax=ax1, wl=wl, data=spd[..., 12:16], color_name="R",
        color=pu.RED, val_list=val_list)
    plot_each_color(
        ax=ax1, wl=wl, data=spd[..., 17:21], color_name="W",
        color='k', val_list=val_list, linestyle='--',
        linewidth=1.5)
    pu.show_and_save(
        fig=fig, legend_loc='upper right', save_fname="./figure/b_spd.png")


def plot_sun_glass_sd():
    spd = load_display_spd(fname="./spd_measure_data/sun_glass.csv")
    wl = spd[..., 0]
    illuminant = spd[..., 1]
    sunglass = spd[..., 2]
    illuminant = illuminant / np.sum(illuminant)
    sunglass = sunglass / np.sum(sunglass)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Spectral distribution",
        graph_title_size=None,
        xlabel="Wavelength [nm]",
        ylabel="Normalized power",
        axis_label_size=None,
        legend_size=12,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=5,
        minor_ytick_num=4)
    ax1.plot(wl, illuminant, label="Illuminant")
    ax1.plot(wl, sunglass, label="Illuminant + Sunglass")
    pu.show_and_save(
        fig=fig, legend_loc='upper right', save_fname="./figure/sunglass.png")


def modify_b_display_spd():
    """
    Extract Red, Green, Blue spectrum and de-noise.
    """
    spd = load_display_spd(fname="./spd_measure_data/rgbw_spd.csv")
    wl = spd[..., 0]
    bb = spd[..., 5]
    gg = spd[..., 10]
    rr = spd[..., 15]
    noise_b = spd[..., 1]
    noise_g = spd[..., 6]
    noise_r = spd[..., 11]

    bb = np.clip(bb - noise_b, 0.0, 1.0)
    gg = np.clip(gg - noise_g, 0.0, 1.0)
    rr = np.clip(rr - noise_r, 0.0, 1.0)
    ww = rr + gg + bb

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Spectral distribution",
        graph_title_size=None,
        xlabel="Wavelength [nm]",
        ylabel="Normalized power",
        axis_label_size=None,
        legend_size=12,
        xlim=None,
        ylim=None,
        # ylim=[-0.005, 0.02],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=5,
        minor_ytick_num=4)
    ax1.plot(wl, rr, color=pu.RED, label='red')
    ax1.plot(wl, gg, color=pu.GREEN, label='green')
    ax1.plot(wl, bb, color=pu.BLUE, label='blue')
    ax1.plot(wl, ww, '--', color='k', label='white', lw=2)
    pu.show_and_save(
        fig=fig, legend_loc='upper right',
        save_fname="./figure/spd_denoise.png")
    print(wl.shape)
    out_spd = tstack([wl, rr, gg, bb, ww])
    np.savetxt(
        "./ref_data/ref_display_spd.csv", out_spd, delimiter=',')


def prepare_display_spd(
        fname="./ref_data/ref_display_spd.csv"):
    data = np.loadtxt(fname=fname, delimiter=",")
    sd = data[..., 1:]
    domain = np.uint16(data[..., 0])
    signals = MultiSignals(data=sd, domain=domain)
    sds = MultiSpectralDistributions(data=signals)

    return sds


def interpolate_festival(
        spd, cmfs, illuminant, spectral_shape=SpectralShape(380, 780, 1)):
    spd2 = spd.interpolate(shape=spectral_shape)
    cmfs2 = cmfs.trim(spectral_shape)
    print(illuminant.values.shape)
    illuminant2 = illuminant.interpolate(shape=spectral_shape)

    return spd2, cmfs2, illuminant2


def calc_display_white_point(
        display_spd_data_fname="./ref_data/ref_display_spd.csv",
        spectral_shape=SpectralShape(380, 780, 1),
        cmfs=CIE1931_CMFS, illuminant=ILLUMINANT_E):
    spd = prepare_display_spd(fname=display_spd_data_fname)
    spd, cmfs, illuminant = interpolate_festival(
        spd=spd, cmfs=cmfs, illuminant=illuminant,
        spectral_shape=spectral_shape)
    rgbw_large_xyz = sd_to_XYZ(
        sd=spd, cmfs=cmfs, illuminant=illuminant)
    rgbw_xyY = XYZ_to_xyY(rgbw_large_xyz)
    # print(rgbw_large_xyz)
    primaries = rgbw_xyY[:3]
    primaries = np.append(primaries, [primaries[0, :]], axis=0)
    white = rgbw_xyY[3]
    print(primaries)
    print(white)
    plot_chromaticity_diagram(primaries, white)


def plot_chromaticity_diagram(
        primaries, white,
        rate=480/755.0*2, xmin=0.0, xmax=0.8, ymin=0.0, ymax=0.9):
    # プロット用データ準備
    # ---------------------------------
    xy_image = tpg.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    cmf_xy = tpg._get_cmfs_xy()

    bt709_gamut, _ = tpg.get_primaries(name=cs.BT709)
    bt2020_gamut, _ = tpg.get_primaries(name=cs.BT2020)
    dci_p3_gamut, _ = tpg.get_primaries(name=cs.P3_D65)
    adoobe_rgb_gamut, _ = tpg.get_primaries(name=cs.ADOBE_RGB)
    xlim = (min(0, xmin), max(0.8, xmax))
    ylim = (min(0, ymin), max(0.9, ymax))

    fig, ax1 = pu.plot_1_graph(
        fontsize=20 * rate,
        figsize=((xmax - xmin) * 10 * rate,
                 (ymax - ymin) * 10 * rate),
        graph_title="CIE1931 Chromaticity Diagram",
        graph_title_size=None,
        xlabel=None, ylabel=None,
        axis_label_size=None,
        legend_size=18 * rate,
        xlim=xlim, ylim=ylim,
        xtick=[x * 0.1 + xmin for x in
               range(int((xlim[1] - xlim[0])/0.1) + 1)],
        ytick=[x * 0.1 + ymin for x in
               range(int((ylim[1] - ylim[0])/0.1) + 1)],
        xtick_size=17 * rate,
        ytick_size=17 * rate,
        linewidth=4 * rate,
        minor_xtick_num=2,
        minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3.5*rate, label=None)
    ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1],
             c=pu.RED, label="BT.709", lw=2.75*rate)
    ax1.plot(bt2020_gamut[:, 0], bt2020_gamut[:, 1],
             c=pu.GREEN, label="BT.2020", lw=2.75*rate)
    ax1.plot(dci_p3_gamut[:, 0], dci_p3_gamut[:, 1],
             c=pu.BLUE, label="DCI-P3", lw=2.75*rate)
    ax1.plot(adoobe_rgb_gamut[:, 0], adoobe_rgb_gamut[:, 1],
             c=pu.SKY, label="AdobeRGB", lw=2.75*rate)
    ax1.plot(
        primaries[..., 0], primaries[..., 1],
        c=pu.ORANGE, label="My Display Primaries", lw=2.75*rate)
    ax1.plot(
        [0.3127], [0.3290], 'x', label='D65', ms=16, mew=5, color='k',
        alpha=0.5)
    ax1.plot(
        white[0], white[1], '+', label='My Display White',
        ms=16, mew=5, color='k')
    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax), alpha=0.5)
    pu.show_and_save(
        fig=fig, legend_loc='upper right',
        save_fname="./figure/display_xy.png")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_measure_patch()
    # plot_b_display_spectral_distribution()
    # plot_sun_glass_sd()
    # modify_b_display_spd()
    calc_display_white_point(
        display_spd_data_fname="./ref_data/ref_display_spd.csv",
        spectral_shape=SpectralShape(380, 780, 1),
        cmfs=CIE1931_CMFS)
    pass
