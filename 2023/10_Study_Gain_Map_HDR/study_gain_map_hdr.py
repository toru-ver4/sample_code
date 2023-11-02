# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import color_space as cs
import plot_utility as pu
import font_control2 as fc2

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


GAIN_MAP_CS_NAME = cs.P3_D65
K_SDR = 10 ** -5
K_HDR = 10 ** -5
GAIN_MAP_ENCODE_GAMMA = 1.0


def apply_gain_map_to_sdr():
    pass


def linearize_input_image(fname, tf_name=tf.ST2084, cs_name=cs.P3_D65):
    img = tpg.img_read_as_float(fname)
    linear_img = tf.eotf_to_luminance(img, tf_name) / tf.REF_WHITE_LUMINANCE
    large_xyz = cs.rgb_to_large_xyz(
        rgb=linear_img, color_space_name=cs_name)
    linear_img = cs.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=GAIN_MAP_CS_NAME)

    return linear_img


def non_linearize_output_image(
        img_linear, tf_name=tf.ST2084, cs_name=cs.P3_D65):
    large_xyz = cs.rgb_to_large_xyz(
        rgb=img_linear, color_space_name=GAIN_MAP_CS_NAME)
    rgb = cs.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=cs_name)
    rgb_non_linear = tf.oetf_from_luminance(
        rgb * tf.REF_WHITE_LUMINANCE, tf_name)

    return rgb_non_linear


def create_gain_map_fname(sdr_fname, hdr_fname):
    """
    Examples
    --------
    >>> sdr_fname = "./img/SDR_TyTP_P3D65.png"
    >>> hdr_fname = "./img/HDR_tyTP_P3D65.png"
    >>> gain_map_fname = create_gain_map_fname(sdr_fname, hdr_fname)
    >>> print(gain_map_fname)
    ./gain_map/gain_map_SDR_TyTP_P3D65_HDR_tyTP_P3D65.png
    """
    pp_sdr = Path(sdr_fname)
    pp_hdr = Path(hdr_fname)

    base_dir = "./gain_map"
    sdr_name = pp_sdr.stem
    hdr_name = pp_hdr.stem

    gain_map_fname = f"{str(base_dir)}/gain_map_{sdr_name}_{hdr_name}.png"

    return gain_map_fname


def create_gain_map_metadata_fname(gain_map_fname):
    return f"{gain_map_fname}_meta.npy"


def create_log2_gain_map_image(sdr_linear, hdr_linear):
    """
    Parameters
    ----------
    sdr_linear : ndarray
        SDR image
    hdr_linear : ndarray
        HDR image
    Note
    ----
    SDR image and HDR image must be the same color gamut.
    """
    def calc_min_val_color(gain_map):
        min_val = [np.min(gain_map[..., idx]) for idx in range(3)]
        return np.array(min_val)

    def calc_max_val_color(gain_map):
        max_val = [np.max(gain_map[..., idx]) for idx in range(3)]
        return np.array(max_val)
    gain_map = np.log2((hdr_linear + K_HDR)/(sdr_linear + K_SDR))
    min_val = calc_min_val_color(gain_map=gain_map)
    max_val = calc_max_val_color(gain_map=gain_map)

    # normalize
    # for idx in range(3):
    #     gain_map[..., idx]\
    #         = (gain_map[..., idx] - min_val[idx])\
    #         / (max_val[idx] - min_val[idx])
    gain_map = (gain_map - min_val) / (max_val - min_val)

    print(f"min_val, max_val = {min_val} {max_val}")
    print(f"gain_map_before_normlize={gain_map[830, 62]}")

    return gain_map, min_val, max_val


def save_gain_map_img(
        gain_map_img_log2, min_val, max_val, gain_map_fname):
    gain_map_metadata_fname = create_gain_map_metadata_fname(
        gain_map_fname=gain_map_fname)

    # apply oetf [option]
    gain_map_img_log2 = gain_map_img_log2 ** (1/GAIN_MAP_ENCODE_GAMMA)

    # save
    tpg.img_wirte_float_as_16bit_int(gain_map_fname, gain_map_img_log2)
    metadata = np.array([min_val, max_val])
    np.save(gain_map_metadata_fname, metadata)


def create_and_save_gain_map(
        sdr_fname="./img/SDR_TyTP_P3D65.png",
        hdr_fname="./img/HDR_tyTP_P3D65.png",
        sdr_cs_name=cs.P3_D65, hdr_cs_name=cs.P3_D65,
        sdr_tf_name=tf.GAMMA24, hdr_tf_name=tf.ST2084):

    gain_map_fname = create_gain_map_fname(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname)

    # linearize
    sdr_liner = linearize_input_image(
        fname=sdr_fname, tf_name=sdr_tf_name, cs_name=sdr_cs_name)
    hdr_liner = linearize_input_image(
        fname=hdr_fname, tf_name=hdr_tf_name, cs_name=hdr_cs_name)

    # create gain map
    gain_map_img_log2, min_val, max_val = create_log2_gain_map_image(
        sdr_linear=sdr_liner, hdr_linear=hdr_liner)

    # save gain map
    save_gain_map_img(
        gain_map_img_log2=gain_map_img_log2,
        min_val=min_val, max_val=max_val, gain_map_fname=gain_map_fname)


def create_intermediate_hdr_image(
        base_sdr_img, gain_map, min_val, max_val, weight_w):
    gg = gain_map * (max_val - min_val) + min_val
    gg = gg * weight_w
    # hdr_img = (base_sdr_img + K_SDR) * (2 ** gg) - K_HDR

    hdr_img = np.zeros_like(base_sdr_img)
    for ii in range(3):
        hdr_img[..., ii]\
            = (base_sdr_img[..., ii] + K_SDR) * (2 ** gg[..., ii]) - K_HDR

    hdr_img[hdr_img < 0] = 0.0

    return hdr_img


def calc_weight_w(hdr_white, sdr_white, min_val, max_val):
    """
    Paramters
    ---------
    hdr_white : float
        HDR peak luminance (e.g. 1000 nits)
    sdr_white : float
        SDR white luminance (e.g. 203 nits)
    min_val : float
        A minimum value of the gain map (log2 space)
    max_val : float
        A maximum value of the gain map (log2 space)
    """
    hh = np.log2(hdr_white/sdr_white)
    ff = (hh - min_val) / (max_val - min_val)

    return np.clip(ff, 0.0, 1.0)


def create_out_hdr_fname_one_file(
        sdr_fname, display_sdr_white_nit, display_hdr_white_nit):
    base_dir = "/work/overuse/2023/10_Gain_Map_HDR"
    basename = Path(sdr_fname).name
    fname = f"{base_dir}/{basename}_{display_sdr_white_nit}_"
    fname += f"{display_hdr_white_nit}.png"

    return fname


def create_out_seq_hdr_fname(
        sdr_fname, idx=0, prefix="SW-100_HW-10000",
        base_dir="/work/overuse/2023/10_Gain_Map_HDR"):

    basename = Path(sdr_fname).name
    fname = f"{base_dir}/{basename}_{prefix}_{idx:04d}.png"

    return fname


def apply_gain_map(
        sdr_fname="./img/SDR_TyTP_P3D65.png",
        sdr_cs_name=cs.P3_D65, sdr_tf_name=tf.GAMMA24,
        hdr_cs_name=cs.P3_D65, hdr_tf_name=tf.ST2084,
        gain_map_img_fname="./img/gain_map_SDR_TyTP_P3D65_HDR_TyTP_P3D65.png",
        display_sdr_white_nit=203, display_hdr_white_nit=10000):
    """
    2種類の出力をするかもしれない。
    1つは正直ベースにパラメータをファイル名に埋め込んだ版。
    もう1つは YouTube アップロード用にパラメータは消して連番ファイルにした版。
    なお、パラメータは文字列として画像に焼く
    """
    metadata_fname = create_gain_map_metadata_fname(gain_map_img_fname)
    metadata = np.load(metadata_fname)
    min_val, max_val = metadata
    # print(f"min_val, max_val = {min_val} {max_val}")

    sdr_liner = linearize_input_image(
        fname=sdr_fname, tf_name=sdr_tf_name, cs_name=sdr_cs_name)
    ww = calc_weight_w(
        hdr_white=display_hdr_white_nit, sdr_white=display_sdr_white_nit,
        min_val=min_val, max_val=max_val)

    gain_map_img = tpg.img_read_as_float(gain_map_img_fname)

    hdr_img_linear = create_intermediate_hdr_image(
        base_sdr_img=sdr_liner, gain_map=gain_map_img, weight_w=ww,
        min_val=min_val, max_val=max_val)

    return hdr_img_linear


def debug_simple_implementation():
    sdr_fname = "./img/SDR_TyTP_P3D65.png"
    hdr_fname = "./img/HDR_tyTP_P3D65.png"
    sdr_cs_name = cs.P3_D65
    hdr_cs_name = cs.P3_D65
    sdr_tf_name = tf.GAMMA24
    hdr_tf_name = tf.ST2084
    create_and_save_gain_map(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname,
        sdr_cs_name=sdr_cs_name, hdr_cs_name=hdr_cs_name,
        sdr_tf_name=sdr_tf_name, hdr_tf_name=hdr_tf_name)

    display_sdr_white_nit = 100
    display_hdr_white_nit = 100
    gain_map_img_fname = create_gain_map_fname(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname)
    hdr_img_linear = apply_gain_map(
        sdr_fname=sdr_fname, gain_map_img_fname=gain_map_img_fname,
        sdr_cs_name=sdr_cs_name, sdr_tf_name=sdr_tf_name,
        hdr_cs_name=hdr_cs_name, hdr_tf_name=hdr_tf_name,
        display_sdr_white_nit=display_sdr_white_nit,
        display_hdr_white_nit=display_hdr_white_nit)
    hdr_img_non_linear = non_linearize_output_image(
        img_linear=hdr_img_linear, tf_name=hdr_tf_name, cs_name=hdr_cs_name)
    fname = create_out_hdr_fname_one_file(
        sdr_fname=sdr_fname,
        display_sdr_white_nit=display_sdr_white_nit,
        display_hdr_white_nit=display_hdr_white_nit)
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, hdr_img_non_linear)


def draw_luminance_info(img, display_sdr_white, display_hdr_white):
    # create instance
    font_color = (0.5, 0.5, 0.5)
    stroke_color = (0.0, 0.0, 0.0)
    text = f"D_SDR_W: {display_sdr_white} nits\n"
    text += f"D_HDR_W: {display_hdr_white} nits\n"
    pos = [60, 30]
    text_draw_ctrl = fc2.TextDrawControl(
        text=text, font_color=font_color,
        font_size=30, font_path=fc2.NOTO_SANS_MONO_REGULAR,
        stroke_width=0, stroke_fill=stroke_color)

    text_draw_ctrl.draw(img=img, pos=pos)


def thread_wrapper_apply_gain_map_seq(args):
    apply_gain_map_seq(**args)


def apply_gain_map_seq(
        idx, sdr_fname, hdr_fname, output_dir,
        sdr_cs_name, hdr_cs_name, sdr_tf_name, hdr_tf_name,
        display_sdr_white_nit, display_hdr_white_nit):
    gain_map_img_fname = create_gain_map_fname(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname)
    hdr_img_linear = apply_gain_map(
        sdr_fname=sdr_fname, gain_map_img_fname=gain_map_img_fname,
        sdr_cs_name=sdr_cs_name, sdr_tf_name=sdr_tf_name,
        hdr_cs_name=hdr_cs_name, hdr_tf_name=hdr_tf_name,
        display_sdr_white_nit=display_sdr_white_nit,
        display_hdr_white_nit=display_hdr_white_nit)
    draw_luminance_info(
        img=hdr_img_linear, display_sdr_white=display_sdr_white_nit,
        display_hdr_white=display_hdr_white_nit)
    hdr_img_non_linear = non_linearize_output_image(
        img_linear=hdr_img_linear, tf_name=hdr_tf_name, cs_name=hdr_cs_name)
    fname = create_out_seq_hdr_fname(
        sdr_fname=sdr_fname, idx=idx, base_dir=output_dir,
        prefix="SDR_White_100_Fixed")
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, hdr_img_non_linear)


def debug_check_effect_of_weight_w():
    sdr_fname = "./img/SDR_TyTP_P3D65.png"
    hdr_fname = "./img/HDR_tyTP_P3D65.png"
    sdr_cs_name = cs.P3_D65
    hdr_cs_name = cs.P3_D65
    sdr_tf_name = tf.GAMMA24
    hdr_tf_name = tf.ST2084
    create_and_save_gain_map(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname,
        sdr_cs_name=sdr_cs_name, hdr_cs_name=hdr_cs_name,
        sdr_tf_name=sdr_tf_name, hdr_tf_name=hdr_tf_name)

    display_sdr_white_nit = 100
    num_of_sample = 512
    # display_hdr_white_nit_list = np.arange(100, 10001, 1)
    display_hdr_white_nit_list = tpg.get_log10_x_scale(
        sample_num=num_of_sample, ref_val=100, min_exposure=0, max_exposure=2)
    display_hdr_white_nit_list\
        = np.round(display_hdr_white_nit_list).astype(np.uint16)

    total_process_num = len(display_hdr_white_nit_list)
    block_process_num = int(cpu_count() / 2 + 0.999)
    block_num = int(round(total_process_num / block_process_num + 0.5))
    output_dir = "/work/overuse/2023/10_Gain_Map_HDR/SDR_100"

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            display_hdr_white_nit = display_hdr_white_nit_list[l_idx]
            d = dict(
                idx=l_idx, sdr_fname=sdr_fname, hdr_fname=hdr_fname,
                output_dir=output_dir,
                sdr_cs_name=sdr_cs_name, hdr_cs_name=hdr_cs_name,
                sdr_tf_name=sdr_tf_name, hdr_tf_name=hdr_tf_name,
                display_sdr_white_nit=display_sdr_white_nit,
                display_hdr_white_nit=display_hdr_white_nit)
            # apply_gain_map_seq(**d)
            args.append(d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_apply_gain_map_seq, args)


def debug_func():
    # debug_simple_implementation()
    debug_check_effect_of_weight_w()



def plot_weighting_parameter_w():
    sdr_white = np.linspace(100, 500, 1024)
    hdr_white = 1000
    gain_map_max_nits = 600
    gain_map_max = np.log2(gain_map_max_nits/100)
    gain_map_min = 0.0

    hh = np.log2(hdr_white / sdr_white)
    ww = (hh - gain_map_min) / gain_map_max - gain_map_min
    ww = np.clip(ww, 0.0, 1.0)

    title = f"HDR white = {hdr_white} nits, "
    title += f"Gain map max = {gain_map_max_nits} nits"

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=title,
        graph_title_size=None,
        xlabel="SDR white [nits]",
        ylabel='Weight parameter "W"',
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(sdr_white, ww, label="W")
    pu.show_and_save(
        fig=fig, legend_loc='upper right', save_fname=None,
        show=True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
    # plot_weighting_parameter_w()
    # min = -2.33710495
    # max = 6.64384191
    # val = 0.26115816
    # print(val * (max - min) + min)
