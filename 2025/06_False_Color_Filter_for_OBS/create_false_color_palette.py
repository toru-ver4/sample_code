# import standard libraries
import os
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np
from scipy.spatial import ConvexHull
from colour.utilities import tstack
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# my libraries
import transfer_functions as tf
from create_gamut_booundary_lut import TyLchLut, is_out_of_gamut_rgb
import color_space as cs
from jzazbz import large_xyz_to_jzazbz, jzazbz_to_large_xyz,\
     jzczhz_to_jzazbz, jzazbz_to_jzczhz
import plot_utility as pu
from make_rgb_signals import make_rgb_signals
import test_pattern_generator2 as tpg


def create_and_plot_ab_plane():
    lut_name = "./lut/JzChz_gb-lut_method_c_ITU-R BT.709_100nits_jj-1024_hh-4096.npy"
    lut = TyLchLut(np.load(lut_name))
    j_max = lut.ll_max
    j_num = 300

    total_process_num = j_num
    block_process_num = int(cpu_count() / 2 + 0.999)
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            j_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={j_idx}")  # User
            if j_idx >= total_process_num:                         # User
                break
            # j_idx = j_num - 1
            d = dict(
                bg_lut_name=lut_name, j_idx=j_idx,
                j_val=j_idx/(j_num-1) * j_max,
                color_space_name=cs.BT709,
                maximum_luminance=100)
            # plot_ab_plane_with_interpolation_core(**d)
            args.append(d)
            # break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_plot_ab_plane_with_interpolation, args)


def thread_wrapper_plot_ab_plane_with_interpolation(args):
    plot_ab_plane_with_interpolation_core(**args)


def create_and_plot_cj_plane():
    lut_name = "./lut/JzChz_gb-lut_method_c_ITU-R BT.709_100nits_jj-1024_hh-4096.npy"
    h_num = 300

    total_process_num = h_num
    block_process_num = int(cpu_count() / 2 + 0.999)
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, h_idx={h_idx}")  # User
            if h_idx >= total_process_num:                         # User
                break
            # j_idx = j_num - 1
            d = dict(
                bg_lut_name=lut_name, h_idx=h_idx,
                h_val=h_idx/(h_num-1)*360,
                color_space_name=cs.BT709,
                maximum_luminance=100)
            # plot_cj_plane_with_interpolation_core(**d)
            args.append(d)
            # break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_plot_cj_plane_with_interpolation, args)


def thread_wrapper_plot_cj_plane_with_interpolation(args):
    plot_cj_plane_with_interpolation_core(**args)


def create_valid_jzazbz_ab_plane_image_sRGB(
        j_val=0.5, ab_max=0.5, ab_sample=512, color_space_name=cs.BT2020,
        bg_rgb_luminance=np.array([50, 50, 50])):
    """
    Create an image that indicates the valid area of the ab plane.

    Parameters
    ----------
    j_val : float
        A Lightness value. range is 0.0 - 1.0
    ab_max : float
        A maximum value of the a, b range.
    ab_sapmle : int
        A number of samples in the image resolution.
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    """
    aa_base = np.linspace(-ab_max, ab_max, ab_sample)
    bb_base = np.linspace(-ab_max, ab_max, ab_sample)
    aa = aa_base.reshape((1, ab_sample))\
        * np.ones_like(bb_base).reshape((ab_sample, 1))
    bb = bb_base.reshape((ab_sample, 1))\
        * np.ones_like(aa_base).reshape((1, ab_sample))
    jj = np.ones_like(aa) * j_val
    jzazbz = np.dstack((jj, aa, bb[::-1])).reshape((ab_sample, ab_sample, 3))
    large_xyz = cs.jzazbz_to_large_xyz(jzazbz)
    rgb_luminance = cs.large_xyz_to_rgb(large_xyz, color_space_name)
    ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance/100)
    rgb_luminance[ng_idx] = bg_rgb_luminance
    rgb_sRGB = tf.oetf_from_luminance(
        np.clip(rgb_luminance, 0.0, 100), tf.SRGB)

    return rgb_sRGB


def create_valid_jzazbz_cj_plane_image_sRGB(
        h_val=50, c_max=1, l_max=1, c_sample=1024, j_sample=1024,
        color_space_name=cs.BT2020, bg_rgb_luminance=np.array([50, 50, 50]),
        maximum_luminance=10000):
    """
    Create an image that indicates the valid area of the ab plane.

    Parameters
    ----------
    h_val : float
        A Hue value. range is 0.0 - 360.0
    c_max : float
        A maximum value of the chroma.
    c_sapmle : int
        A number of samples for the chroma.
    l_sample : int
        A number of samples for the lightness.
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    bg_lightness : float
        background lightness value.
    maximum_luminance : float
        maximum luminance of the target display device.
    """
    cc_base = np.linspace(0, c_max, c_sample)
    jj_base = np.linspace(0, l_max, j_sample)
    cc = cc_base.reshape(1, c_sample)\
        * np.ones_like(jj_base).reshape(j_sample, 1)
    jj = jj_base.reshape(j_sample, 1)\
        * np.ones_like(cc_base).reshape(1, c_sample)
    hh = np.ones_like(cc) * h_val

    jczhz = np.dstack([jj[::-1], cc, hh]).reshape((j_sample, c_sample, 3))
    jzazbz = jzczhz_to_jzazbz(jczhz)
    large_xyz = jzazbz_to_large_xyz(jzazbz)
    rgb_luminance = cs.large_xyz_to_rgb(large_xyz, color_space_name)
    ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance/maximum_luminance)

    rgb_luminance[ng_idx] = bg_rgb_luminance

    rgb_sRGB = tf.oetf(
        np.clip(rgb_luminance/maximum_luminance, 0.0, 1), tf.SRGB)

    return rgb_sRGB


def plot_ab_plane_with_interpolation_core(
        bg_lut_name, j_idx, j_val, color_space_name, maximum_luminance):
    if maximum_luminance <= 101:
        ab_max = 0.20
    elif maximum_luminance <= 1001:
        ab_max = 0.30
    else:
        ab_max = 0.40

    ab_sample = 1536
    hue_sample = 1536
    bg_lut = TyLchLut(np.load(bg_lut_name))
    rgb_sRGB = create_valid_jzazbz_ab_plane_image_sRGB(
        j_val=j_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([50, 50, 50])
    )
    
    hh_base = np.linspace(0, 360, hue_sample)
    jj_base = np.ones_like(hh_base) * j_val
    jh_array = tstack([jj_base, hh_base])
    jzczhz = bg_lut.interpolate(lh_array=jh_array)
    chroma = jzczhz[..., 1]
    hue = np.deg2rad(jzczhz[..., 2])
    aa = chroma * np.cos(hue)
    bb = chroma * np.sin(hue)

    jzazbz_luminance = jzczhz_to_jzazbz(jzczhz[0])
    large_xyz = jzazbz_to_large_xyz(jzazbz_luminance)
    luminance = large_xyz[1]

    graph_title = f"azbz plane,  {color_space_name},  "
    graph_title += f"Jz={j_val:.2f},  Luminance={luminance:.2f} nits"
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 12),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=graph_title,
        graph_title_size=None,
        xlabel="az", ylabel="bz",
        axis_label_size=None,
        legend_size=17,
        xlim=[-ab_max, ab_max],
        ylim=[-ab_max, ab_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=1,
        minor_xtick_num=None,
        minor_ytick_num=None)
    
    ax1.imshow(
        rgb_sRGB, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    ax1.plot(aa, bb, color='k')
    fname = "/work/overuse/2025/06_obs_false_color/jzazbz_debug/"
    fname += f"azbz_w_lut_{color_space_name}_"
    fname += f"{maximum_luminance}nits_{j_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)
    

def plot_cj_plane_with_interpolation_core(
        bg_lut_name, h_idx, h_val, color_space_name, maximum_luminance):
    bg_lut = TyLchLut(lut=np.load(bg_lut_name))
    sample_num = 1536
    jj_sample = 1536
    if maximum_luminance <= 101:
        if color_space_name == cs.BT709:
            cc_max = 0.17
            jj_max = 0.18
        elif color_space_name == cs.P3_D65:
            cc_max = 0.18
            jj_max = 0.18
        else:
            cc_max = 0.22
            jj_max = 0.18
    elif maximum_luminance <= 1001:
        if color_space_name == cs.BT709:
            cc_max = 0.27
            jj_max = 0.43
        elif color_space_name == cs.P3_D65:
            cc_max = 0.30
            jj_max = 0.43
        else:
            cc_max = 0.37
            jj_max = 0.43
    else:
        if color_space_name == cs.BT709:
            cc_max = 0.35
            jj_max = 1.0
        elif color_space_name == cs.P3_D65:
            cc_max = 0.36
            jj_max = 1.0
        else:
            cc_max = 0.45
            jj_max = 1.0
    print(f"h_val={h_val} started")

    rgb_st2084 = create_valid_jzazbz_cj_plane_image_sRGB(
        h_val=h_val, c_max=cc_max, l_max=jj_max,
        c_sample=sample_num, j_sample=sample_num,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([50, 50, 50]),
        maximum_luminance=maximum_luminance)
    graph_title = f"CzJz plane,  {color_space_name},  hue={h_val:.2f}°,  "
    graph_title += f"target={maximum_luminance} nits"

    jj_base = np.linspace(0, bg_lut.ll_max, jj_sample)
    hh_base = np.ones_like(jj_base) * h_val
    jh_array = tstack([jj_base, hh_base])
    # jzczhz = get_gamut_boundary_lch_from_lut(
    #     lut=bg_lut, lh_array=jh_array, lightness_max=1.0)
    jzczhz = bg_lut.interpolate(lh_array=jh_array)

    chroma = jzczhz[..., 1]
    lightness = jzczhz[..., 0]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 12),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=graph_title,
        graph_title_size=None,
        xlabel="Cz", ylabel="Jz",
        axis_label_size=None,
        legend_size=17,
        xlim=[0, cc_max],
        ylim=[0, jj_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=1.5,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_st2084, extent=(0, cc_max, 0, jj_max), aspect='auto')
    ax1.plot(chroma, lightness, color='k')
    fname = "/work/overuse/2025/06_obs_false_color/jzazbz_debug/"
    fname += f"CzJz_w_lut_{color_space_name}_{maximum_luminance}-nits_"
    fname += f"{h_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)
    

def create_jzazbz_boundary_data(rgb, n=11):
    rgb_linear = rgb ** 2.2
    large_xyz = cs.rgb_to_large_xyz(rgb_linear, cs.BT709) * 100
    jzazbz = large_xyz_to_jzazbz(large_xyz)
    
    return jzazbz
    

def plot_color_volume():
    n = 40
    rgb = make_rgb_signals(n=n, max_val=1023) / 1023
    jzazbz = create_jzazbz_boundary_data(rgb, n=n)
    Jz, az, bz = jzazbz[:,0], jzazbz[:,1], jzazbz[:,2]

    # プロット準備
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    colors = np.clip(rgb, 0, 1)
    ax.scatter(az, bz, Jz, c=colors, marker='o', s=10)

    # 軸ラベル
    ax.set_xlabel('az')
    ax.set_ylabel('bz')
    ax.set_zlabel('Jz')
    ax.set_title('Color Volume in Jzazbz Space')
    ax.view_init(elev=90, azim=-90)

    # レイアウト調整＆表示
    plt.tight_layout()
    plt.show()


def plot_cj_plane_with_hue(h_val):
    bg_lut = TyLchLut(lut=np.load("./lut/JzChz_gb-lut_method_c_ITU-R BT.709_100nits_jj-1024_hh-4096.npy"))
    sample_num = 1536
    jj_sample = 1536
    cc_max = 0.17
    jj_max = 0.18
    maximum_luminance = 100

    rgb_st2084 = create_valid_jzazbz_cj_plane_image_sRGB(
        h_val=h_val, c_max=cc_max, l_max=jj_max,
        c_sample=sample_num, j_sample=sample_num,
        color_space_name=cs.BT709,
        bg_rgb_luminance=np.array([50, 50, 50]),
        maximum_luminance=maximum_luminance)
    graph_title = f"CzJz plane,  {cs.BT709},  hue={h_val:.2f}°,  "
    graph_title += f"target={maximum_luminance} nits"

    jj_base = np.linspace(0, bg_lut.ll_max, jj_sample)
    hh_base = np.ones_like(jj_base) * h_val
    jh_array = tstack([jj_base, hh_base])
    # jzczhz = get_gamut_boundary_lch_from_lut(
    #     lut=bg_lut, lh_array=jh_array, lightness_max=1.0)
    jzczhz = bg_lut.interpolate(lh_array=jh_array)

    chroma = jzczhz[..., 1]
    lightness = jzczhz[..., 0]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 12),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=graph_title,
        graph_title_size=None,
        xlabel="Cz", ylabel="Jz",
        axis_label_size=None,
        legend_size=17,
        xlim=[0, cc_max],
        ylim=[0, jj_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=1.5,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_st2084, extent=(0, cc_max, 0, jj_max), aspect='auto')
    ax1.plot(chroma, lightness, color='k')
    fname = f"./img/CzJz_Plane_{h_val:.1f}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def calculate_hue_for_rgb(rgb):
    rgb_linear = tf.eotf(rgb, tf.SRGB)
    large_xyz = cs.rgb_to_large_xyz(rgb_linear, cs.BT709) * 100
    jzazbz = large_xyz_to_jzazbz(large_xyz)
    jzczhz = jzazbz_to_jzczhz(jzazbz)
    hz = jzczhz[2]
    print(f"RGB: {rgb}, Hue: {hz:.3f} degree")


def calculate_hue_for_color_palette():
    color_list = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],    
        [1, 1, 0], [0, 1, 1], [1, 0, 1],
    ]
    for color in color_list:
        calculate_hue_for_rgb(rgb=color)


def create_pallate_image(hue, num_of_sample=4, idx=0):
    rgb_linear = create_color_palette_each_hue(hue=hue, num_of_sample=num_of_sample)
    x_org = np.linspace(0, 1, num_of_sample)
    x_new = np.linspace(0, 1, 1600)
    rgb_intp = np.zeros((1600, 3))

    for c_idx in range(3):
        rgb_intp[:, c_idx] = np.interp(x_new, x_org, rgb_linear[:, c_idx])

    rgb_intp_srgb = tf.oetf(rgb_intp, tf.SRGB)
    img = tpg.h_color_line_to_img(rgb_intp_srgb, 96)
    tpg.img_wirte_float_as_16bit_int(
        f"./img/hue-{idx}_sample-{num_of_sample:04d}.png", img
    )


def create_color_palette(num_of_sample=6):
    # hue_list = [
    #     #  B        C        G        Y       O       R        M
    #     258.812, 203.825, 132.719, 101.804, 72.126, 42.477, 320.769
    # ]
    hue_list = [
        #  B        C        G        Y       O       R        M
        254.000, 203.825, 132.719, 101.804, 72.126, 42.477, 320.769
    ]
    linear_color_palette_list_bt709 = []
    for idx, hue in enumerate(hue_list):
        create_pallate_image(hue, num_of_sample=num_of_sample, idx=idx)
        create_pallate_image(hue, num_of_sample=1600, idx=idx)
        rgb_linear_bt709 = create_color_palette_each_hue(hue=hue, num_of_sample=num_of_sample)
        linear_color_palette_list_bt709.append(rgb_linear_bt709)
        # plot_cj_plane_with_hue(hue)

    # convert from bt.709 to bt.2020
    rgb_709 = np.array(linear_color_palette_list_bt709)
    large_xyz = cs.rgb_to_large_xyz(rgb_709, cs.BT709)
    rgb_2020 = cs.large_xyz_to_rgb(large_xyz, cs.BT2020)
    linear_color_paletter_list_bt2020 = rgb_2020


    # write to .effect file
    effect_str = ""
    for p_idx, palette in enumerate(linear_color_paletter_list_bt2020):
        effect_str += f"// palette_{p_idx}\n"
        for entry_idx, row in enumerate(palette):
            effect_str += "uniform float3 p{}_{} = {{ {:.6f}, {:.6f}, {:.6f} }};\n".format(
                p_idx, entry_idx, row[0], row[1], row[2]
            )
        effect_str += "\n"

    # Write to "paletter.effect" in the same directory.
    output_filename = "flase_color_palette.effect"
    with open(output_filename, "w") as f:
        f.write(effect_str)

    # write to .py file
    py_effect_str = "# Python palette definitions.\n"
    py_effect_str = "# This color palette is for BT.2100-PQ color space.\n"
    for idx, rgb_linear in enumerate(linear_color_paletter_list_bt2020):
        py_effect_str += f"palette_{idx} = [\n"
        for row in rgb_linear:
            py_effect_str += "    [{:.6f}, {:.6f}, {:.6f}],\n".format(row[0], row[1], row[2])
        py_effect_str += "]\n\n"

    output_filename_py = "flase_color_palette.py"
    with open(output_filename_py, "w") as f:
        f.write(py_effect_str)


def create_color_palette_each_hue(hue, num_of_sample=1024):
    lut_data = np.load("./lut/JzChz_gb-lut_method_c_ITU-R BT.709_100nits_jj-1024_hh-4096.npy")
    lut = TyLchLut(lut_data)
    cusp = lut.get_cusp(hue)

    j_ed = cusp[0]
    c_ed = cusp[1]
    h_ed = cusp[2]

    # j_st = j_ed / 3.0
    j_st = j_ed - 0.035
    c_st = c_ed / 2.0
    h_st = h_ed

    # Create linear interpolated arrays for each channel
    j_samples = np.linspace(j_st, j_ed, num=num_of_sample)
    c_samples = np.linspace(c_st, c_ed, num=num_of_sample)
    h_samples = np.linspace(h_st, h_ed, num=num_of_sample)

    # Stack into a single ndarray with shape (num_of_sample, 3)
    jzczhz = np.stack((j_samples, c_samples, h_samples), axis=-1)
    jzazbz = jzczhz_to_jzazbz(jzczhz)
    large_xyz = jzazbz_to_large_xyz(jzazbz) / 100.0
    rgb_linear = cs.large_xyz_to_rgb(large_xyz, cs.BT709)

    return np.clip(rgb_linear, 0.0, 1.0)


def concat_color_palette_img(num_of_sample=6):
    img_buf = []
    for idx in range(7):
        fname = f"./img/hue-{idx}_sample-{num_of_sample:04d}.png"
        img_buf.append(tpg.img_read_as_float(fname))

    img = np.vstack(img_buf)
    tpg.img_wirte_float_as_16bit_int(
        f"./img/concat_sample-{num_of_sample:04d}.png", img
    )

    img_buf = []
    for idx in range(7):
        fname = f"./img/hue-{idx}_sample-1600.png"
        img_buf.append(tpg.img_read_as_float(fname))

    img = np.vstack(img_buf)
    tpg.img_wirte_float_as_16bit_int(
        "./img/concat_sample-1600.png", img
    )


def plot_czjz_plane_for_blog_core(idx, hue):
    bg_lut = TyLchLut(lut=np.load("./lut/JzChz_gb-lut_method_c_ITU-R BT.709_100nits_jj-1024_hh-4096.npy"))
    num_of_palette_sample = 6
    sample_num = 1536
    jj_sample = 1536
    cc_max = 0.17
    jj_max = 0.18
    maximum_luminance = 100

    cusp = bg_lut.get_cusp(hue)

    j_ed = cusp[0]
    c_ed = cusp[1]

    # j_st = j_ed / 3.0
    j_st = j_ed - 0.035
    c_st = c_ed / 2.0

    # Create linear interpolated arrays for each channel
    jj = np.linspace(j_st, j_ed, num=num_of_palette_sample)
    cc = np.linspace(c_st, c_ed, num=num_of_palette_sample)

    rgb_st2084 = create_valid_jzazbz_cj_plane_image_sRGB(
        h_val=hue, c_max=cc_max, l_max=jj_max,
        c_sample=sample_num, j_sample=sample_num,
        color_space_name=cs.BT709,
        bg_rgb_luminance=np.array([50, 50, 50]),
        maximum_luminance=maximum_luminance)
    graph_title = f"CzJz plane,  {cs.BT709},  hue={hue:.2f}°,  "
    graph_title += f"target={maximum_luminance} nits"

    jj_base = np.linspace(0, bg_lut.ll_max, jj_sample)
    hh_base = np.ones_like(jj_base) * hue
    jh_array = tstack([jj_base, hh_base])
    # jzczhz = get_gamut_boundary_lch_from_lut(
    #     lut=bg_lut, lh_array=jh_array, lightness_max=1.0)
    jzczhz = bg_lut.interpolate(lh_array=jh_array)

    chroma = jzczhz[..., 1]
    lightness = jzczhz[..., 0]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=graph_title,
        graph_title_size=20,
        xlabel="Cz", ylabel="Jz",
        axis_label_size=None,
        legend_size=17,
        xlim=[0, cc_max],
        ylim=[0, jj_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=1.5,
        minor_xtick_num=None,
        minor_ytick_num=None)
    for x in np.arange(0.00, 0.17, 0.02):
        ax1.axvline(x=x, color='k', linestyle='--', linewidth=0.5, zorder=3, alpha=0.15)
    for y in np.arange(0.00, 0.19, 0.02):
        ax1.axhline(y=y, color='k', linestyle='--', linewidth=0.5, zorder=3, alpha=0.15)
    ax1.imshow(
        rgb_st2084, extent=(0, cc_max, 0, jj_max), aspect='auto', zorder=2
    )
    ax1.plot(chroma, lightness, color='k', zorder=3)
    ax1.plot(cc, jj, "o",
             markerfacecolor='none',
             markeredgecolor='k',
             markersize=24,
             markeredgewidth=2,
             zorder=3)
    ax1.plot(cc, jj, "o",
             markerfacecolor='none',
             markeredgecolor='w',
             markersize=28,
             markeredgewidth=2,
             zorder=3)
    fname = f"./img/color_palette_{idx}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def plot_czjz_plane_for_blog_ctrl():
    hue_list = [
        #  B        C        G        Y       O       R        M
        254.000, 203.825, 132.719, 101.804, 72.126, 42.477, 320.769
    ]

    for idx, hue in enumerate(hue_list):
        plot_czjz_plane_for_blog_core(idx=idx, hue=hue)


def concat_czjz_plane():
    h_img_buf = []
    v_img_buf = []
    dummy_img = np.ones((1000, 1000, 3))
    idx = 0
    for v_idx in range(3):
        h_img_buf = []
        for h_idx in range(3):
            fname = f"./img/color_palette_{idx}.png"
            if os.path.exists(fname):
                temp_img = tpg.img_read_as_float(fname)
                temp_img = temp_img[..., :3]  # remove alpha channel
            else:
                temp_img = dummy_img

            # draw border
            border_color = [0] * temp_img.shape[2]
            temp_img[0, :] = border_color
            temp_img[-1, :] = border_color
            temp_img[:, 0] = border_color
            temp_img[:, -1] = border_color

            h_img_buf.append(temp_img)
            idx += 1
        v_img_buf.append(np.hstack(h_img_buf))

    img = np.vstack(v_img_buf)
    tpg.img_wirte_float_as_16bit_int("./img/color_palette_concat.png", img)


def debug():
    # create_and_plot_ab_plane()
    # create_and_plot_cj_plane()
    # plot_color_volume()
    # calculate_hue_for_color_palette()
    # plot_czjz_plane_for_blog_ctrl()
    concat_czjz_plane()
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug()
    # num_of_palette_sample = 6
    # create_color_palette(num_of_sample=num_of_palette_sample)
    # concat_color_palette_img(num_of_sample=num_of_palette_sample)
