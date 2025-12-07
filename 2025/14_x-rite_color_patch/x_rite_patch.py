import os

import numpy as np
from colour.continuous import MultiSignals
from colour import (
    MultiSpectralDistributions,
    SpectralShape,
    SDS_ILLUMINANTS,
    MSDS_CMFS,
    xy_to_XYZ,
    XYZ_to_xy,
    sd_to_XYZ,
    Lab_to_XYZ,
)
from colour.utilities import tstack, tsplit
from colour.difference import delta_E_CIE2000
from colour.adaptation import chromatic_adaptation_VonKries
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import string
from coolpi.image.colourchecker import ColourCheckerSpectral

import color_space as cs
import transfer_functions as tf
import plot_utility as pu

ILLUMINANT_D65 = SDS_ILLUMINANTS['D65']
CIE1931_CMFS = MultiSpectralDistributions(MSDS_CMFS["cie_2_1931"])

CONST_CIELAB_DELTA = 6.0/29.0


def _func_t(t):
    threshold = CONST_CIELAB_DELTA ** 3
    upper = (t > threshold) * (t ** (1/3))
    lower = (t <= threshold) * (t/(3 * (CONST_CIELAB_DELTA ** 2)) + 4/29)
    return upper + lower


def _func_t_inverse(t):
    upper = (t > CONST_CIELAB_DELTA) * (t ** 3)
    lower = (t <= CONST_CIELAB_DELTA) * 3 * (CONST_CIELAB_DELTA ** 2) * (t - 4/29)
    return upper + lower


def ty_lab_to_large_xyz(lab, white=[95.047, 100.000, 108.883]):
    l, a, b = tsplit(lab)
    white = [x / white[1] for x in white]
    large_x = white[0] * _func_t_inverse((l + 16)/116 + a/500)
    large_y = white[1] * _func_t_inverse((l + 16)/116)
    large_z = white[2] * _func_t_inverse((l + 16)/116 - b/200)

    return tstack((large_x, large_y, large_z))


def ty_large_xyz_to_lab(large_xyz, white=[95.047, 100.000, 108.883]):
    x, y, z = tsplit(large_xyz)
    white = [x / white[1] for x in white]
    l = 116 * _func_t(y/white[1]) - 16
    a = 500 * (_func_t(x/white[0]) - _func_t(y/white[1]))
    b = 200 * (_func_t(y/white[1]) - _func_t(z/white[2]))

    return tstack((l, a, b))


def trim_and_iterpolate(x, spectral_shape):
    y = x.trim(shape=spectral_shape)
    y = y.interpolate(shape=spectral_shape)

    return y


def trim_and_interpolate_in_advance(
        spd, cmfs, illuminant,
        spectral_shape=SpectralShape(380, 730, 1)):
    spd2 = trim_and_iterpolate(spd, spectral_shape)
    cmfs2 = trim_and_iterpolate(cmfs, spectral_shape)
    illuminant2 = trim_and_iterpolate(illuminant, spectral_shape)

    return spd2, cmfs2, illuminant2


def load_cdsg_spectrum_data():
    """
    Returns
    -------
    wavelength = data[0]
    spectrum = data[1:]
    """
    fname = "./data/ColorCheckerDigitalSG_Spectrum.txt"
    usecols = [0] + [5 + x for x in range(36)]

    data = np.loadtxt(fname, skiprows=14, usecols=usecols)
    num_of_wavelength = len(data[0]) - 1
    wavelength = [380 + x * 10 for x in range(num_of_wavelength)]

    # print(wavelength)
    data = data[..., 1:]
    print(data.shape)
    data = conv_v_base_idx_to_h_base_idx(data, num_of_v=10, num_of_h=14).T

    ref_data = np.ones((num_of_wavelength, 1))
    # print(data.shape)
    # print(ref_data.shape)
    data = np.append(data, ref_data, axis=1)
    # print(data)

    signals = MultiSignals(data=data, domain=wavelength)
    sds = MultiSpectralDistributions(data=signals)

    # print(sds)

    return sds


def calc_large_xyz_from_sds(
        sds: MultiSpectralDistributions, spectral_shape=SpectralShape(380, 730, 10)):
    illuminanct_d65 = ILLUMINANT_D65
    cie1931_cmfs = CIE1931_CMFS
    sd, cmfs, illuminant = trim_and_interpolate_in_advance(
        spd=sds, cmfs=cie1931_cmfs, illuminant=illuminanct_d65,
        spectral_shape=spectral_shape)
    
    large_xyz = sd_to_XYZ(sd=sd, cmfs=cmfs, illuminant=illuminant)

    return large_xyz / 100


def plot_color_checker_sg(
        rgb, num_of_h=14, num_of_v=10,
        patch_margin=0.08,
        fig_width=10,
        save_fname=None):

    # パッチの一辺（セル1マス=1 の座標系）
    patch_size = 1.0 - 2 * patch_margin

    # 左右の外側マージン：パッチサイズの 1/2 相当
    outer_margin_x = patch_size * 0.65 - patch_margin

    # 上下の外側マージン：少し広めに取りたいので係数だけ大きく
    #   例: 0.75倍 → お好みで調整してください
    outer_margin_y = patch_size * 0.65 - patch_margin

    # --- ここがポイント：データの縦横比から figsize を決める ---
    data_width = num_of_h + 2 * outer_margin_x
    data_height = num_of_v + 2 * outer_margin_y

    fig_height = fig_width * (data_height / data_width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # -------- パッチ描画 --------
    for v_idx in range(num_of_v):
        for h_idx in range(num_of_h):
            idx = v_idx * num_of_h + h_idx
            color = rgb[idx]

            x = h_idx + patch_margin
            y = v_idx + patch_margin

            ax.add_patch(
                Rectangle((x, y), patch_size, patch_size,
                          facecolor=color, edgecolor="none")
            )

    # グリッド＋外側マージンをそのまま xlim/ylim に
    ax.set_xlim(-outer_margin_x, num_of_h + outer_margin_x)
    ax.set_ylim(num_of_v + outer_margin_y, -outer_margin_y)  # 上を 0 行目にするため反転

    # -------- ラベル描画 --------
    font_color = (0.7, 0.7, 0.7)
    pe_stroke = [pe.Stroke(linewidth=2, foreground="black"), pe.Normal()]

    # 行番号（1..num_of_v）
    for v_idx in range(num_of_v):
        y = v_idx + 0.5  # パッチ中心
        ax.text(-outer_margin_x * 0.33, y, str(v_idx + 1),
                va="center", ha="center",
                color=font_color, fontsize=10, fontweight="bold",
                path_effects=pe_stroke, zorder=10)

    # 列アルファベット（A..）
    letters = string.ascii_uppercase[:num_of_h]
    for h_idx, letter in enumerate(letters):
        x = h_idx + 0.5  # パッチ中心
        ax.text(x, num_of_v + outer_margin_y * 0.33, letter,
                va="center", ha="center",
                color=font_color, fontsize=10, fontweight="bold",
                path_effects=pe_stroke, zorder=10)

    # 軸領域を図いっぱいに
    ax.set_position([0, 0, 1, 1])

    # 図全体の枠
    fig.add_artist(
        Rectangle(
            (0, 0), 1, 1,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor=(0.7, 0.7, 0.7),
            linewidth=1
        )
    )

    if save_fname is not None:
        plt.savefig(
            save_fname,
            facecolor=fig.get_facecolor(),
            dpi=100,
            bbox_inches="tight",
            pad_inches=0.01,
        )

    plt.show()


def check_ccdsg_before_nov_2014_spectrum_data():
    sds = load_cdsg_spectrum_data()
    large_xyz = calc_large_xyz_from_sds(sds=sds)
    print(large_xyz[0])
    srgb_linear = cs.large_xyz_to_rgb(large_xyz, color_space_name=cs.sRGB)
    srgb = tf.oetf(np.clip(srgb_linear, 0.0, 1.0), tf.SRGB)
    plot_color_checker_sg(rgb=srgb[:140], save_fname="./img/babel_color_before_nov_2014.png")


def get_ccdsg_data_from_coolpi(checker_name="CCDSG"):
    ccobj = ColourCheckerSpectral(checker_name)
    patch_id_list = list(ccobj.patches.keys())
    wavelength_range = ccobj.nm_range
    wavelength_interval = ccobj.nm_interval
    num_of_wavelength = (wavelength_range[1] - wavelength_range[0]) // wavelength_interval + 1
    sr_data = np.zeros((num_of_wavelength, len(patch_id_list)))
    wavelength = np.arange(
        wavelength_range[0], wavelength_range[1] + wavelength_interval, wavelength_interval
    )
    for idx, patch_id in enumerate(patch_id_list):
        sr = ccobj.get_patch_lambda_values(patch_id)
        sr_data[:, idx] = sr
    sr_data /= 100.0
    sr_data = conv_v_base_idx_to_h_base_idx(sr_data.T, num_of_v=10, num_of_h=14).T
    ref_data = np.ones((num_of_wavelength, 1))
    sr_data = np.append(sr_data, ref_data, axis=1)

    signals = MultiSignals(data=sr_data, domain=wavelength)
    sds = MultiSpectralDistributions(data=signals)

    return sds


def check_ccdsg_spectrum_data(checker_name="CCDSG"):
    ccdsg_sds = get_ccdsg_data_from_coolpi(checker_name=checker_name)
    wl_st, wl_ed = int(ccdsg_sds.wavelengths[0]), int(ccdsg_sds.wavelengths[-1])
    large_xyz = calc_large_xyz_from_sds(
        sds=ccdsg_sds, spectral_shape=SpectralShape(wl_st, wl_ed, 1)
    )
    srgb_linear = cs.large_xyz_to_rgb(large_xyz, color_space_name=cs.sRGB)
    srgb = tf.oetf(np.clip(srgb_linear, 0.0, 1.0), tf.SRGB)
    plot_color_checker_sg(rgb=srgb[:140], save_fname=f"./img/{checker_name}.png")


def debug_plot_single_patch(
        wavelength : np.ndarray,
        sr : np.ndarray,
        line_color : np.ndarray | None = None,
        save_fname : str | None = None):
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel=None,
        ylabel=None,
        axis_label_size=None,
        legend_size=17,
        xlim=[370, 740],
        ylim=[0.0, 1.0],
        xtick=[400 + 50 * x for x in range(7)],
        ytick=[x * 0.1 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    line_color = line_color if line_color is not None else 'k'
    ax1.plot(wavelength, sr, '-', color=line_color)
    show = True if save_fname is None else False
    pu.show_and_save(fig=fig, legend_loc=None, save_fname=save_fname, show=show)


def debug_plot_dual_patch(
        wl1 : np.ndarray,
        sr1 : np.ndarray,
        wl2 : np.ndarray,
        sr2 : np.ndarray,
        line_color : np.ndarray | None = None,
        save_fname : str | None = None):
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=None,
        graph_title_size=None,
        xlabel=None,
        ylabel=None,
        axis_label_size=None,
        legend_size=17,
        xlim=[370, 740],
        ylim=[0.0, 1.0],
        xtick=[400 + 50 * x for x in range(7)],
        ytick=[x * 0.1 for x in range(11)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    line_color = line_color if line_color is not None else 'k'
    ax1.plot(wl1, sr1, '-', color=line_color)
    ax1.plot(wl2, sr2, '--', color=line_color)
    show = True if save_fname is None else False
    pu.show_and_save(fig=fig, legend_loc=None, save_fname=save_fname, show=show)


def debug_plot_single_patch_spectrum():
    sds = load_cdsg_spectrum_data()
    wavelength = sds.wavelengths
    sr = sds.values[..., 0]

    debug_plot_single_patch(wavelength=wavelength, sr=sr)


def debug_plot_dual_patch_spectrum():
    spectral_shape = SpectralShape(380, 730, 10)
    patch_idx = 2
    spd_1 = load_cdsg_spectrum_data()
    spd_1 = trim_and_iterpolate(spd_1, spectral_shape)
    wl_1 = spd_1.wavelengths
    sr_1 = spd_1.values[..., patch_idx]

    spd_2 = get_ccdsg_data_from_coolpi(checker_name="XRCCSG")
    spd_2 = trim_and_iterpolate(spd_2, spectral_shape)  
    wl_2 = spd_2.wavelengths
    sr_2 = spd_2.values[..., patch_idx]

    debug_plot_dual_patch(wl1=wl_1, sr1=sr_1, wl2=wl_2, sr2=sr_2)


def debug_plot_dual_patch_spectrum_all():
    spectral_shape = SpectralShape(380, 730, 10)

    spd_1 = load_cdsg_spectrum_data()
    spd_1 = trim_and_iterpolate(spd_1, spectral_shape)
    wl_1 = spd_1.wavelengths
    spd_1_sr = spd_1.values

    spd_2 = get_ccdsg_data_from_coolpi(checker_name="XRCCSG")
    spd_2 = trim_and_iterpolate(spd_2, spectral_shape)  
    wl_2 = spd_2.wavelengths
    spd_2_sr = spd_2.values

    large_xyz = calc_large_xyz_from_sds(sds=spd_1)
    srgb_linear = cs.large_xyz_to_rgb(large_xyz, color_space_name=cs.sRGB)
    srgb = tf.oetf(np.clip(srgb_linear, 0.0, 1.0), tf.SRGB)

    num_of_vertical, num_of_horizontal = 10, 14

    fig, axes = plt.subplots(
        num_of_vertical, num_of_horizontal,
        figsize=(num_of_horizontal * 1.5, num_of_vertical * 1.5),
        sharex=False, sharey=False
    )

    for v_idx in range(num_of_vertical):
        for h_idx in range(num_of_horizontal):
            idx = h_idx * num_of_vertical + v_idx
            ax = axes[v_idx][h_idx]

            ax.set_facecolor((*srgb[idx], 0.3))

            # 2 本の分光特性を重ねて描画
            ax.plot(wl_1, spd_1_sr[..., idx], '-', lw=1.5)
            ax.plot(wl_2, spd_2_sr[..., idx], '--', lw=1.5)

            ax.set_ylim(0.0, 1.0)
            ax.tick_params(labelsize=6)
            ax.grid(True, color="#cccccc", linewidth=0.5, linestyle='--')

    fig.tight_layout()
    plt.show()


def load_xrite_official_ccdsg_xyz_value(kind='after'):
    if kind == 'before':
        fname = "./data/ColorCheckerSG_Before_Nov2014.txt"
        skip_rows = 10
    elif kind == "after":
        fname = "./data/ColorCheckerSG_After_Nov2014.txt"
        skip_rows = 15
    else:
        raise ValueError("invalid kind type")
    
    lab = np.loadtxt(fname, skiprows=skip_rows, usecols=(1, 2, 3))
    lab = conv_v_base_idx_to_h_base_idx(lab, num_of_h=14, num_of_v=10)

    large_xyz = Lab_to_XYZ(lab, illuminant=cs.D50)

    d50_xyz = xy_to_XYZ(cs.D50)
    d65_xyz = xy_to_XYZ(cs.D65)

    print(f"before_bradford = {large_xyz[0]}, {large_xyz[10]}, {large_xyz[11]}, {large_xyz[20]}")
    large_xyz = chromatic_adaptation_VonKries(large_xyz, d50_xyz, d65_xyz, transform="Bradford")

    return large_xyz


def check_xrite_threoretical_value(kind="after"):
    large_xyz = load_xrite_official_ccdsg_xyz_value(kind=kind)
    rgb = cs.large_xyz_to_rgb(large_xyz, cs.BT709)
    srgb = tf.oetf(np.clip(rgb, 0.0, 1.0), tf.SRGB)

    plot_color_checker_sg(rgb=srgb)


def load_displayhdr_patch_xyz():
    fname = "./data/DisplayHDR_96_Pacth.txt"
    data = np.loadtxt(fname, skiprows=1, delimiter=",", usecols=(0, 7, 8, 9))
    idx = data[:96, 0].astype(np.uint8) - 1
    large_xyz = data[:96, 1:] / 100.0
    
    return idx, large_xyz


def debug_load_displayhdr_patch():
    idx, large_xyz = load_displayhdr_patch_xyz()
    print(len(idx))
    # print(large_xyz)


def conv_v_base_idx_to_h_base_idx(src_data, num_of_v=10, num_of_h=14):
    dst_data = src_data.reshape(num_of_h, num_of_v, -1)
    dst_data = np.transpose(dst_data, (1, 0, 2))
    dst_data = dst_data.reshape(num_of_v * num_of_h, -1)

    return dst_data


def extract_96_patch_from_140_patch(x: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    x : np.ndarray
        A tristimulus value. The last number of the shape must be 3.
    """
    idx, _ = load_displayhdr_patch_xyz()
    return x[idx]


def compare_xrite_lab_and_displayhdr():
    xyz_from_lab = load_xrite_official_ccdsg_xyz_value()
    xyz_from_lab = extract_96_patch_from_140_patch(xyz_from_lab)

    _, displayhdr_xyz = load_displayhdr_patch_xyz()

    for idx in range(len(xyz_from_lab)):
        xrite = xyz_from_lab[idx, 1]
        displayhdr = displayhdr_xyz[idx, 1]
        diff = displayhdr - xrite
        print(f"idx={idx}, X-Rite={xrite:.4f}, DisplayHDR={displayhdr:.4f}, diff={diff:.4f}")


def plot_96_patch_of_140_patch():
    xyz_from_lab = load_xrite_official_ccdsg_xyz_value()

    rgb = cs.large_xyz_to_rgb(xyz_from_lab, cs.BT709)
    rgb_srgb = tf.oetf(np.clip(rgb, 0.0, 1.0), tf.SRGB)

    plot_96_patch_of_140_patch_core(rgb=rgb_srgb, save_fname="./img/96_patch_of_140_patch.png")


def plot_96_patch_of_140_patch_core(
        rgb, num_of_h=14, num_of_v=10, patch_margin=0.08, fig_width=10, save_fname=None):
    
    ref_idx, _ = load_displayhdr_patch_xyz()

    # パッチの一辺（セル1マス=1 の座標系）
    patch_size = 1.0 - 2 * patch_margin

    # 左右の外側マージン：パッチサイズの 1/2 相当
    outer_margin_x = patch_size * 0.65 - patch_margin

    # 上下の外側マージン：少し広めに取りたいので係数だけ大きく
    #   例: 0.75倍 → お好みで調整してください
    outer_margin_y = patch_size * 0.65 - patch_margin

    # --- ここがポイント：データの縦横比から figsize を決める ---
    data_width = num_of_h + 2 * outer_margin_x
    data_height = num_of_v + 2 * outer_margin_y

    fig_height = fig_width * (data_height / data_width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    width = 1 - 2 * patch_margin
    height = 1 - 2 * patch_margin

    # -------- パッチ描画 --------
    for v_idx in range(num_of_v):
        for h_idx in range(num_of_h):
            idx = v_idx * num_of_h + h_idx
            color = rgb[idx]

            x = h_idx + patch_margin
            y = v_idx + patch_margin

            ax.add_patch(
                Rectangle((x, y), patch_size, patch_size,
                          facecolor=color, edgecolor="none")
            )
            if idx not in ref_idx:
                x0, x1 = x + width * 0.2, x + width * 0.8
                y0, y1 = y + height * 0.2, y + height * 0.8
                effects = [pe.Stroke(linewidth=4, foreground="black"), pe.Normal()]
                ax.plot([x0, x1], [y0, y1], color="white", lw=2, zorder=5, path_effects=effects)
                ax.plot([x0, x1], [y1, y0], color="white", lw=2, zorder=5, path_effects=effects)

    # グリッド＋外側マージンをそのまま xlim/ylim に
    ax.set_xlim(-outer_margin_x, num_of_h + outer_margin_x)
    ax.set_ylim(num_of_v + outer_margin_y, -outer_margin_y)  # 上を 0 行目にするため反転

    # -------- ラベル描画 --------
    font_color = (0.7, 0.7, 0.7)
    pe_stroke = [pe.Stroke(linewidth=2, foreground="black"), pe.Normal()]

    # 行番号（1..num_of_v）
    for v_idx in range(num_of_v):
        y = v_idx + 0.5  # パッチ中心
        ax.text(-outer_margin_x * 0.33, y, str(v_idx + 1),
                va="center", ha="center",
                color=font_color, fontsize=10, fontweight="bold",
                path_effects=pe_stroke, zorder=10)

    # 列アルファベット（A..）
    letters = string.ascii_uppercase[:num_of_h]
    for h_idx, letter in enumerate(letters):
        x = h_idx + 0.5  # パッチ中心
        ax.text(x, num_of_v + outer_margin_y * 0.33, letter,
                va="center", ha="center",
                color=font_color, fontsize=10, fontweight="bold",
                path_effects=pe_stroke, zorder=10)

    # 軸領域を図いっぱいに
    ax.set_position([0, 0, 1, 1])

    # 図全体の枠
    fig.add_artist(
        Rectangle(
            (0, 0), 1, 1,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor=(0.7, 0.7, 0.7),
            linewidth=1
        )
    )

    if save_fname is not None:
        plt.savefig(
            save_fname,
            facecolor=fig.get_facecolor(),
            dpi=100,
            bbox_inches="tight",
            pad_inches=0.01,
        )

    plt.show()


def plot_display_hdr_96_xyz_patch(
        num_of_h=14, num_of_v=10, patch_margin=0.08, fig_width=10, save_fname=None):
    
    ref_idx, large_xyz = load_displayhdr_patch_xyz()
    rgb = cs.large_xyz_to_rgb(large_xyz, cs.BT709)
    rgb_srgb = tf.oetf(np.clip(rgb, 0.0, 1.0), tf.SRGB)

    # パッチの一辺（セル1マス=1 の座標系）
    patch_size = 1.0 - 2 * patch_margin

    # 左右の外側マージン：パッチサイズの 1/2 相当
    outer_margin_x = patch_size * 0.65 - patch_margin

    # 上下の外側マージン：少し広めに取りたいので係数だけ大きく
    #   例: 0.75倍 → お好みで調整してください
    outer_margin_y = patch_size * 0.65 - patch_margin

    # --- ここがポイント：データの縦横比から figsize を決める ---
    data_width = num_of_h + 2 * outer_margin_x
    data_height = num_of_v + 2 * outer_margin_y

    fig_height = fig_width * (data_height / data_width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    width = 1 - 2 * patch_margin
    height = 1 - 2 * patch_margin

    # -------- パッチ描画 --------
    p_idx = 0
    for v_idx in range(num_of_v):
        for h_idx in range(num_of_h):
            idx = v_idx * num_of_h + h_idx
            if idx in ref_idx:
                color = rgb_srgb[p_idx]
                p_idx += 1
            else:
                color = [0.2, 0.2, 0.2]

            x = h_idx + patch_margin
            y = v_idx + patch_margin

            ax.add_patch(
                Rectangle((x, y), patch_size, patch_size,
                          facecolor=color, edgecolor="none")
            )
            if idx not in ref_idx:
                x0, x1 = x + width * 0.2, x + width * 0.8
                y0, y1 = y + height * 0.2, y + height * 0.8
                effects = [pe.Stroke(linewidth=4, foreground="black"), pe.Normal()]
                ax.plot([x0, x1], [y0, y1], color="white", lw=2, zorder=5, path_effects=effects)
                ax.plot([x0, x1], [y1, y0], color="white", lw=2, zorder=5, path_effects=effects)

    # グリッド＋外側マージンをそのまま xlim/ylim に
    ax.set_xlim(-outer_margin_x, num_of_h + outer_margin_x)
    ax.set_ylim(num_of_v + outer_margin_y, -outer_margin_y)  # 上を 0 行目にするため反転

    # -------- ラベル描画 --------
    font_color = (0.7, 0.7, 0.7)
    pe_stroke = [pe.Stroke(linewidth=2, foreground="black"), pe.Normal()]

    # 行番号（1..num_of_v）
    for v_idx in range(num_of_v):
        y = v_idx + 0.5  # パッチ中心
        ax.text(-outer_margin_x * 0.33, y, str(v_idx + 1),
                va="center", ha="center",
                color=font_color, fontsize=10, fontweight="bold",
                path_effects=pe_stroke, zorder=10)

    # 列アルファベット（A..）
    letters = string.ascii_uppercase[:num_of_h]
    for h_idx, letter in enumerate(letters):
        x = h_idx + 0.5  # パッチ中心
        ax.text(x, num_of_v + outer_margin_y * 0.33, letter,
                va="center", ha="center",
                color=font_color, fontsize=10, fontweight="bold",
                path_effects=pe_stroke, zorder=10)

    # 軸領域を図いっぱいに
    ax.set_position([0, 0, 1, 1])

    # 図全体の枠
    fig.add_artist(
        Rectangle(
            (0, 0), 1, 1,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor=(0.7, 0.7, 0.7),
            linewidth=1
        )
    )

    if save_fname is not None:
        plt.savefig(
            save_fname,
            facecolor=fig.get_facecolor(),
            dpi=100,
            bbox_inches="tight",
            pad_inches=0.01,
        )

    plt.show()


def research_display_hdr_pacth_luminance():
    valid_idx, displayhdr_xyz = load_displayhdr_patch_xyz()

    # X-Rite data
    xrite_xyz = load_xrite_official_ccdsg_xyz_value()
    xrite_xyz = xrite_xyz[valid_idx]

    # COOLPI data
    ccdsg_sds = get_ccdsg_data_from_coolpi(checker_name="CCDSG")
    wl_st, wl_ed = int(ccdsg_sds.wavelengths[0]), int(ccdsg_sds.wavelengths[-1])
    coolpi_xyz = calc_large_xyz_from_sds(
        sds=ccdsg_sds, spectral_shape=SpectralShape(wl_st, wl_ed, 1)
    )
    coolpi_xyz = coolpi_xyz[valid_idx]

    displayhdr_y = displayhdr_xyz[..., 1]
    xrite_y = xrite_xyz[..., 1]
    coolpi_y = coolpi_xyz[..., 1]
    diff_with_xrite = displayhdr_y - xrite_y
    diff_with_coolpi = displayhdr_y - coolpi_y

    x = np.arange(len(valid_idx))
    abs_max = np.max(np.abs([diff_with_xrite, diff_with_coolpi]))
    pad = 1.05 * abs_max if abs_max > 0 else 0.5  # 0 対策でデフォルト幅を確保

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(18, 6),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Title",
        graph_title_size=None,
        xlabel="DisplayHDR Patch Index",
        ylabel="Difference [nits]",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=(-pad, pad),  # 0 を中心に上下対称
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=2,
        minor_xtick_num=None,
        minor_ytick_num=None
    )
    bar_w = 0.3
    ax1.bar(x - bar_w/2, diff_with_xrite, width=bar_w, label="Diff vs X-Rite LAB")
    ax1.bar(x + bar_w/2, diff_with_coolpi, width=bar_w, label="Diff vs COOLPI Spectrum Data")

    y_lumi = (1 - displayhdr_y/np.max(displayhdr_y)) * abs_max
    ax1.plot(x, y_lumi, '--ok', label="1 - original_luminance (Normalized)")
    pu.show_and_save(fig=fig, legend_loc='lower right', save_fname=None, show=True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # check_ccdsg_before_nov_2014_spectrum_data()
    # check_ccdsg_spectrum_data(checker_name="CCDSG")
    # check_ccdsg_spectrum_data(checker_name="XRCCSG")
    # check_xrite_threoretical_value(kind="after")

    # debug_plot_single_patch_spectrum()
    # debug_plot_dual_patch_spectrum()
    # debug_plot_dual_patch_spectrum_all()

    # load_xrite_official_ccdsg_xyz_value(kind="before")
    # load_xrite_official_ccdsg_xyz_value(kind="after")

    # debug_load_displayhdr_patch()
    # compare_xrite_lab_and_displayhdr()

    # plot_96_patch_of_140_patch()
    # plot_display_hdr_96_xyz_patch(save_fname="./img/DisplayHDR_96_patch.png")

    research_display_hdr_pacth_luminance()
