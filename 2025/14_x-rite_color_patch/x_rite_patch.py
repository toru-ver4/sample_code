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
import test_pattern_generator2 as tpg

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


def add_icc_profile_to_image(src_fname: str, dst_fname_suffix: str, icc_profile_path: str):
    """
    Save ``src_fname`` to a new file in the same folder with ``dst_fname_suffix``
    inserted before the extension and ``icc_profile_path`` embedded.

    Parameters
    ----------
    src_fname : str
        Source image path.
    dst_fname_suffix : str
        Suffix appended to the source filename (before the extension) to form
        the output filename.
    icc_profile_path : str
        ICC profile file path.
    """
    if not os.path.isfile(src_fname):
        raise FileNotFoundError(f"source image not found: {src_fname}")
    if not os.path.isfile(icc_profile_path):
        raise FileNotFoundError(f"icc profile not found: {icc_profile_path}")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - only hit when Pillow is missing
        raise ImportError("Pillow is required to embed an ICC profile.") from exc

    with open(icc_profile_path, "rb") as f:
        icc_bytes = f.read()

    src_dir, src_basename = os.path.split(src_fname)
    src_stem, src_ext = os.path.splitext(src_basename)
    dst_fname = os.path.join(src_dir, f"{src_stem}{dst_fname_suffix}{src_ext}")

    with Image.open(src_fname) as img:
        save_kwargs = img.info.copy()
        save_kwargs.pop("icc_profile", None)  # replace existing profile
        if "exif" in img.info:
            save_kwargs["exif"] = img.info["exif"]
        save_kwargs["icc_profile"] = icc_bytes

        img.save(dst_fname, format=img.format, **save_kwargs)


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
        rgb_bt2020, num_of_h=14, num_of_v=10,
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
            color = rgb_bt2020[idx]

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
        add_icc_profile_to_image(
            save_fname, dst_fname_suffix="_with_profile", icc_profile_path="./img/sRGB_BT2020.icc"
        )

    plt.show()


def check_ccdsg_before_nov_2014_spectrum_data():
    sds = load_cdsg_spectrum_data()
    large_xyz = calc_large_xyz_from_sds(sds=sds)
    print(large_xyz[0])
    bt2020_linear = cs.large_xyz_to_rgb(large_xyz, color_space_name=cs.BT2020)
    bt2020 = tf.oetf(np.clip(bt2020_linear, 0.0, 1.0), tf.SRGB)
    plot_color_checker_sg(rgb_bt2020=bt2020[:140], save_fname="./img/babel_color_before_nov_2014.png")


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
    large_xyz = load_coolpi_xyz_value(checker_name=checker_name)
    bt2020_linear = cs.large_xyz_to_rgb(large_xyz, color_space_name=cs.BT2020)
    bt2020 = tf.oetf(np.clip(bt2020_linear, 0.0, 1.0), tf.SRGB)
    plot_color_checker_sg(rgb_bt2020=bt2020[:140], save_fname=f"./img/{checker_name}.png")


def load_coolpi_xyz_value(checker_name="CCDSG"):
    ccdsg_sds = get_ccdsg_data_from_coolpi(checker_name=checker_name)
    wl_st, wl_ed = int(ccdsg_sds.wavelengths[0]), int(ccdsg_sds.wavelengths[-1])
    large_xyz = calc_large_xyz_from_sds(
        sds=ccdsg_sds, spectral_shape=SpectralShape(wl_st, wl_ed, 1)
    )

    return large_xyz


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
            idx = v_idx * num_of_horizontal + h_idx
            ax = axes[v_idx][h_idx]

            ax.set_facecolor((*srgb[idx], 0.5))

            # 2 本の分光特性を重ねて描画
            ax.plot(wl_1, spd_1_sr[..., idx], '-', color=pu.RED, lw=1.5)
            ax.plot(wl_2, spd_2_sr[..., idx], '--', color=pu.BLUE, lw=1.5)

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

    plot_color_checker_sg(rgb_bt2020=srgb, save_fname=f"./img/ColorChecker_Digital_SG_X-Rite_Official_{kind}_Nov_2024.png")


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
    # xyz_from_lab = load_xrite_official_ccdsg_xyz_value()
    large_xyz = load_coolpi_xyz_value()

    rgb = cs.large_xyz_to_rgb(large_xyz, cs.BT709)
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
                ax.plot([x0, x1], [y0, y1], color=pu.PINK, lw=2, zorder=5, path_effects=effects)
                ax.plot([x0, x1], [y1, y0], color=pu.PINK, lw=2, zorder=5, path_effects=effects)

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


def plot_cc_18_patch_xy():
    # idx, display_hdr_xyz = load_displayhdr_patch_xyz()
    cc_idx_140 = [
        18, 19, 20, 21, 22, 23,
        32, 33, 34, 35, 36, 37,
        46, 47, 48, 49, 50, 51,
    ]
    cc_idx_96 = [
        6, 7, 8, 9, 10, 11,
        18, 19, 20, 21, 22, 23,
        30, 31, 32, 33, 34, 35
    ]
    ccdsg_official_xyz = load_xrite_official_ccdsg_xyz_value()[cc_idx_140]
    ccdsg_official_xyY = XYZ_to_xy(ccdsg_official_xyz)

    ccdsg_coolpi_xyz = load_coolpi_xyz_value()[cc_idx_140]
    ccdsg_coolpi_xyY = XYZ_to_xy(ccdsg_coolpi_xyz)

    _, displayhdr_xyz = load_displayhdr_patch_xyz()
    displayhdr_xyz = displayhdr_xyz[cc_idx_96]
    displayhdr_xyY = XYZ_to_xy(displayhdr_xyz)

    cc_clasic_xyY = tpg.get_color_checker_xyY_value()[:18]

    rgb = tf.oetf(np.clip(cs.large_xyz_to_rgb(ccdsg_coolpi_xyz, cs.BT709), 0.0, 1.0), tf.SRGB)

    rate = 1.3
    xmin = -0.1
    xmax = 0.8
    ymin = -0.1
    ymax = 1.0
    # プロット用データ準備
    # ---------------------------------
    st_wl = 380
    ed_wl = 780
    wl_step = 1
    plot_wl_list = [
        410, 450, 470, 480, 485, 490, 495,
        500, 505, 510, 520, 530, 540, 550, 560, 570, 580, 590,
        600, 620, 690]
    cmf_xy = pu.calc_horseshoe_chromaticity(
        st_wl=st_wl, ed_wl=ed_wl, wl_step=wl_step)
    cmf_xy_norm = pu.calc_normal_pos(
        xy=cmf_xy, normal_len=0.05, angle_degree=90)
    wl_list = np.arange(st_wl, ed_wl + 1, wl_step)
    xy_image = pu.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cmf_xy=cmf_xy)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20 * rate,
        figsize=((xmax - xmin) * 10 * rate,
                 (ymax - ymin) * 10 * rate),
        graph_title="ColorChecker 18 Patch",
        graph_title_size=None,
        xlabel=None, ylabel=None,
        axis_label_size=None,
        legend_size=14 * rate,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        xtick=[x * 0.1 + xmin for x in
               range(int((xmax - xmin)/0.1) + 1)],
        ytick=[x * 0.1 + ymin for x in
               range(int((ymax - ymin)/0.1) + 1)],
        xtick_size=17 * rate,
        ytick_size=17 * rate,
        linewidth=4 * rate,
        minor_xtick_num=2,
        minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=2*rate, label=None)
    ax1.scatter(
        ccdsg_official_xyY[..., 0], ccdsg_official_xyY[..., 1], c=rgb, marker='o', s=60, label="ColorChecker Digital SG X-Rite Official Data"
    )
    ax1.scatter(
        ccdsg_coolpi_xyY[..., 0], ccdsg_coolpi_xyY[..., 1], c=rgb, marker='s', s=100, label="ColorChecker Digital SG COOLPI Spectrum Data"
    )

    # background CC 24
    ax1.scatter(cc_clasic_xyY[..., 0], cc_clasic_xyY[..., 1], c='black', marker='x', s=120, lw=3, zorder=1)
    # foreground Cc 24
    ax1.scatter(
        cc_clasic_xyY[..., 0], cc_clasic_xyY[..., 1], c=rgb, marker='x', s=80, lw=1, zorder=2, label='ColorChecker 24 - After November 2014'
    )

    # background DisplayHDR
    ax1.scatter(displayhdr_xyY[..., 0], displayhdr_xyY[..., 1], c='black', marker='+', s=160, lw=3, zorder=1)
    # foreground DisplayHDR
    ax1.scatter(
        displayhdr_xyY[..., 0], displayhdr_xyY[..., 1], c=rgb, marker='+', s=100, lw=1, zorder=2, label='DisplayHDR'
    )

    for idx, wl in enumerate(wl_list):
        if wl not in plot_wl_list:
            continue
        pu.draw_wl_annotation(
            ax1=ax1, wl=wl, rate=rate,
            st_pos=[cmf_xy_norm[idx, 0], cmf_xy_norm[idx, 1]],
            ed_pos=[cmf_xy[idx, 0], cmf_xy[idx, 1]])
    bt709_gamut = pu.get_primaries(name=cs.BT709)
    ax1.plot(
        bt709_gamut[:, 0], bt709_gamut[:, 1], '-o', ms=5,
        markeredgecolor='black', markeredgewidth=1.0, c=pu.BLUE, label="BT.709", lw=1.25*rate
    )
    bt2020_gamut = pu.get_primaries(name=cs.BT2020)
    ax1.plot(
        bt2020_gamut[:, 0], bt2020_gamut[:, 1], '-o', ms=5,
        markeredgecolor='black', markeredgewidth=1.0, c=pu.RED, label="BT.2020", lw=1.25*rate
    )
    dci_p3_gamut = pu.get_primaries(name=cs.P3_D65)
    ax1.plot(
        dci_p3_gamut[:, 0], dci_p3_gamut[:, 1], '-o', ms=5,
        markeredgecolor='black', markeredgewidth=1.0, c=pu.GREEN, label="DCI-P3", lw=1.25*rate
    )
    ax1.plot(
        [0.3127], [0.3290], 'x', label='D65', ms=12*rate, mew=2*rate,
        color='k', alpha=0.8)
    # ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax), alpha=0.5)
    pu.show_and_save(
        fig=fig, legend_loc='upper right', fontsize=14,
        save_fname="./img/chromaticity_18_patch.png", show=True
    )


def plot_cc_96_patch_xy():
    idx_96_of_140, display_hdr_xyz = load_displayhdr_patch_xyz()
    displayhdr_xyY = XYZ_to_xy(display_hdr_xyz)[1:]

    ccdsg_official_xyz = load_xrite_official_ccdsg_xyz_value()[idx_96_of_140]
    ccdsg_official_xyY = XYZ_to_xy(ccdsg_official_xyz)[1:]

    rgb = tf.oetf(np.clip(cs.large_xyz_to_rgb(display_hdr_xyz[1:], cs.BT709), 0.0, 1.0), tf.SRGB)

    rate = 1.3
    xmin = -0.1
    xmax = 0.8
    ymin = -0.1
    ymax = 1.0
    # プロット用データ準備
    # ---------------------------------
    st_wl = 380
    ed_wl = 780
    wl_step = 1
    plot_wl_list = [
        410, 450, 470, 480, 485, 490, 495,
        500, 505, 510, 520, 530, 540, 550, 560, 570, 580, 590,
        600, 620, 690]
    cmf_xy = pu.calc_horseshoe_chromaticity(
        st_wl=st_wl, ed_wl=ed_wl, wl_step=wl_step)
    cmf_xy_norm = pu.calc_normal_pos(
        xy=cmf_xy, normal_len=0.05, angle_degree=90)
    wl_list = np.arange(st_wl, ed_wl + 1, wl_step)
    xy_image = pu.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cmf_xy=cmf_xy)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20 * rate,
        figsize=((xmax - xmin) * 10 * rate,
                 (ymax - ymin) * 10 * rate),
        graph_title="ColorChecker 96 Patch",
        graph_title_size=None,
        xlabel=None, ylabel=None,
        axis_label_size=None,
        legend_size=14 * rate,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        xtick=[x * 0.1 + xmin for x in
               range(int((xmax - xmin)/0.1) + 1)],
        ytick=[x * 0.1 + ymin for x in
               range(int((ymax - ymin)/0.1) + 1)],
        xtick_size=17 * rate,
        ytick_size=17 * rate,
        linewidth=4 * rate,
        minor_xtick_num=2,
        minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=2*rate, label=None)

    ax1.scatter(
        ccdsg_official_xyY[..., 0], ccdsg_official_xyY[..., 1], c=rgb, marker='o', s=60, label="ColorChecker Digital SG X-Rite Official Data"
    )

    # background DisplayHDR
    ax1.scatter(displayhdr_xyY[..., 0], displayhdr_xyY[..., 1], c='black', marker='+', s=160, lw=3, zorder=1)
    # foreground DisplayHDR
    ax1.scatter(
        displayhdr_xyY[..., 0], displayhdr_xyY[..., 1], c=rgb, marker='+', s=100, lw=1, zorder=2, label='DisplayHDR'
    )

    for idx_96_of_140, wl in enumerate(wl_list):
        if wl not in plot_wl_list:
            continue
        pu.draw_wl_annotation(
            ax1=ax1, wl=wl, rate=rate,
            st_pos=[cmf_xy_norm[idx_96_of_140, 0], cmf_xy_norm[idx_96_of_140, 1]],
            ed_pos=[cmf_xy[idx_96_of_140, 0], cmf_xy[idx_96_of_140, 1]])
    bt709_gamut = pu.get_primaries(name=cs.BT709)
    ax1.plot(
        bt709_gamut[:, 0], bt709_gamut[:, 1], '-o', ms=5,
        markeredgecolor='black', markeredgewidth=1.0, c=pu.BLUE, label="BT.709", lw=1.25*rate
    )
    bt2020_gamut = pu.get_primaries(name=cs.BT2020)
    ax1.plot(
        bt2020_gamut[:, 0], bt2020_gamut[:, 1], '-o', ms=5,
        markeredgecolor='black', markeredgewidth=1.0, c=pu.RED, label="BT.2020", lw=1.25*rate
    )
    dci_p3_gamut = pu.get_primaries(name=cs.P3_D65)
    ax1.plot(
        dci_p3_gamut[:, 0], dci_p3_gamut[:, 1], '-o', ms=5,
        markeredgecolor='black', markeredgewidth=1.0, c=pu.GREEN, label="DCI-P3", lw=1.25*rate
    )
    ax1.plot(
        [0.3127], [0.3290], 'x', label='D65', ms=12*rate, mew=2*rate,
        color='k', alpha=0.8)
    # ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax), alpha=0.5)
    pu.show_and_save(
        fig=fig, legend_loc='upper right', fontsize=14,
        save_fname="./img/chromaticity_96_patch.png", show=True
    )


def plot_cc_96_patch_ab_plane():
    idx_96_of_140, display_hdr_xyz = load_displayhdr_patch_xyz()

    ccdsg_official_xyz = load_xrite_official_ccdsg_xyz_value()[idx_96_of_140]
    ccdsg_official_xyY = XYZ_to_xy(ccdsg_official_xyz)

    rgb = tf.oetf(np.clip(cs.large_xyz_to_rgb(display_hdr_xyz[1:], cs.BT709), 0.0, 1.0), tf.SRGB)


def plot_chromaticity_data_all():
    # plot_cc_18_patch_xy()
    # plot_cc_96_patch_xy()
    plot_cc_96_patch_ab_plane()
    

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    check_ccdsg_before_nov_2014_spectrum_data()
    check_ccdsg_spectrum_data(checker_name="CCDSG")
    check_ccdsg_spectrum_data(checker_name="XRCCSG")
    # check_xrite_threoretical_value(kind="before")
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

    # research_display_hdr_pacth_luminance()
    # plot_chromaticity_data_all()
