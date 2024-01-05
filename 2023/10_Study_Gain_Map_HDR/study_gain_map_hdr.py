# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np
from colour.io import read_image
from colour import read_LUT

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import color_space as cs
import plot_utility as pu
import font_control2 as fc2
from tonemapping import create_ty_tonemap_v2_func

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
HDR_REF_WHITE = 203


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

    def save_diff(img_1, img_2, fname):
        diff = (img_1 - img_2) * tf.REF_WHITE_LUMINANCE
        img = tf.oetf_from_luminance(np.clip(diff, 0.0, 10000))
        tpg.img_wirte_float_as_16bit_int(fname, img)

    save_diff(hdr_linear, sdr_linear, "./debug/00_hdr_sdr.png")
    save_diff(sdr_linear, hdr_linear, "./debug/00_sdr_hdr.png")

    return gain_map, min_val, max_val


def save_gain_map_img(
        gain_map_fname, gain_map_img_log2, min_val, max_val,
        min_hdr_capacity, max_hdr_capacity):
    gain_map_metadata_fname = create_gain_map_metadata_fname(
        gain_map_fname=gain_map_fname)

    # apply oetf [option]
    gain_map_img_log2 = gain_map_img_log2 ** (1/GAIN_MAP_ENCODE_GAMMA)

    # save
    tpg.img_wirte_float_as_16bit_int(gain_map_fname, gain_map_img_log2)
    print(min_val, max_val, min_hdr_capacity, max_hdr_capacity)
    metadata = np.array([
        min_val[0], min_val[1], min_val[2],
        max_val[0], max_val[1], max_val[2],
        min_hdr_capacity, max_hdr_capacity])
    np.save(gain_map_metadata_fname, metadata)


def create_and_save_gain_map(
        sdr_fname="./img/SDR_TyTP_P3D65.png",
        hdr_fname="./img/HDR_tyTP_P3D65.png",
        sdr_white=HDR_REF_WHITE,
        sdr_cs_name=cs.P3_D65, hdr_cs_name=cs.P3_D65,
        sdr_tf_name=tf.GAMMA24, hdr_tf_name=tf.ST2084,
        min_hdr_capacity=np.log2(203/HDR_REF_WHITE),
        max_hdr_capacity=np.log2(1000/HDR_REF_WHITE)):

    gain_map_fname = create_gain_map_fname(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname)

    # linearize sdr image
    sdr_liner = linearize_input_image(
        fname=sdr_fname, tf_name=sdr_tf_name, cs_name=sdr_cs_name)
    # convert from 100 nits to 203 nits for consistency
    sdr_liner = sdr_liner / tf.REF_WHITE_LUMINANCE * sdr_white

    # linearlize hdr image
    hdr_liner = linearize_input_image(
        fname=hdr_fname, tf_name=hdr_tf_name, cs_name=hdr_cs_name)

    # create gain map
    gain_map_img_log2, min_val, max_val = create_log2_gain_map_image(
        sdr_linear=sdr_liner, hdr_linear=hdr_liner)

    # save gain map
    save_gain_map_img(
        gain_map_img_log2=gain_map_img_log2,
        min_val=min_val, max_val=max_val, gain_map_fname=gain_map_fname,
        min_hdr_capacity=min_hdr_capacity, max_hdr_capacity=max_hdr_capacity)


def create_intermediate_hdr_image(
        base_sdr_img, gain_map, min_val, max_val, weight_w):
    gg = gain_map * (max_val - min_val) + min_val
    gg = gg * weight_w
    print(f"weight_w={weight_w}")

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
        A minimum value of the HDR capacity (log2 space)
    max_val : float
        A maximum value of the HDR capacity (log2 space)
    """
    hh = np.log2(hdr_white/sdr_white)
    ff = (hh - min_val) / (max_val - min_val)

    return np.clip(ff, 0.0, 1.0)


def create_out_hdr_fname_one_file(
        hdr_fname, display_sdr_white_nit, display_hdr_white_nit):
    base_dir = "/work/overuse/2023/10_Gain_Map_HDR"
    basename = Path(hdr_fname).name
    dst_dir_path = Path(f"{base_dir}/{basename}")
    dst_dir_path.mkdir(parents=True, exist_ok=True)
    fname = f"{base_dir}/{basename}/{basename}_{display_sdr_white_nit}_"
    fname += f"{display_hdr_white_nit}.png"

    return fname


def create_out_seq_hdr_fname(
        sdr_fname, idx=0, prefix="SW-100_HW-10000",
        base_dir="/work/overuse/2023/10_Gain_Map_HDR"):

    basename = Path(sdr_fname).stem
    dst_dir_path = Path(f"{base_dir}/{basename}")
    dst_dir_path.mkdir(parents=True, exist_ok=True)
    fname = f"{base_dir}/{basename}/{basename}_{prefix}_{idx:04d}.png"

    return fname


def load_gain_map_meatadata(metadata_fname):
    metadata = np.load(metadata_fname)
    min_val = metadata[0:3]
    max_val = metadata[3:6]
    min_hdr_capacity = metadata[6]
    max_hdr_capacity = metadata[7]

    return min_val, max_val, min_hdr_capacity, max_hdr_capacity


def apply_gain_map(
        sdr_fname="./img/SDR_TyTP_P3D65.png",
        sdr_cs_name=cs.P3_D65, sdr_tf_name=tf.GAMMA24,
        gain_map_img_fname="./img/gain_map_SDR_TyTP_P3D65_HDR_TyTP_P3D65.png",
        display_sdr_white_nit=203, display_hdr_white_nit=10000):
    """
    2種類の出力をするかもしれない。
    1つは正直ベースにパラメータをファイル名に埋め込んだ版。
    もう1つは YouTube アップロード用にパラメータは消して連番ファイルにした版。
    なお、パラメータは文字列として画像に焼く
    """
    metadata_fname = create_gain_map_metadata_fname(gain_map_img_fname)
    min_val, max_val, min_hdr_capacity, max_hdr_capacity\
        = load_gain_map_meatadata(metadata_fname=metadata_fname)
    print(f"min_val, max_val = {min_val} {max_val}")
    # print(f"min_hdr_capacity = {min_hdr_capacity}")
    # print(f"max_hdr_capacity = {max_hdr_capacity}")

    sdr_liner = linearize_input_image(
        fname=sdr_fname, tf_name=sdr_tf_name, cs_name=sdr_cs_name)
    sdr_liner = sdr_liner * display_sdr_white_nit / tf.REF_WHITE_LUMINANCE

    ww = calc_weight_w(
        hdr_white=display_hdr_white_nit, sdr_white=display_sdr_white_nit,
        min_val=min_hdr_capacity, max_val=max_hdr_capacity)

    gain_map_img = tpg.img_read_as_float(gain_map_img_fname)

    hdr_img_linear = create_intermediate_hdr_image(
        base_sdr_img=sdr_liner, gain_map=gain_map_img, weight_w=ww,
        min_val=min_val, max_val=max_val)

    return hdr_img_linear


def debug_simple_implementation(
        sdr_fname="./img/SDR_TyTP_P3D65.png",
        hdr_fname="./img/HDR_tyTP_P3D65.png",
        min_hdr_capacity=np.log2(203/HDR_REF_WHITE),
        max_hdr_capacity=np.log2(1000/HDR_REF_WHITE),
        display_sdr_white_nit=203,
        display_hdr_white_nit=10000):
    sdr_cs_name = cs.P3_D65
    hdr_cs_name = cs.P3_D65
    sdr_tf_name = tf.GAMMA24
    hdr_tf_name = tf.ST2084

    create_and_save_gain_map(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname,
        sdr_cs_name=sdr_cs_name, hdr_cs_name=hdr_cs_name,
        sdr_tf_name=sdr_tf_name, hdr_tf_name=hdr_tf_name,
        min_hdr_capacity=min_hdr_capacity, max_hdr_capacity=max_hdr_capacity)

    gain_map_img_fname = create_gain_map_fname(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname)
    hdr_img_linear = apply_gain_map(
        sdr_fname=sdr_fname, gain_map_img_fname=gain_map_img_fname,
        sdr_cs_name=sdr_cs_name, sdr_tf_name=sdr_tf_name,
        display_sdr_white_nit=display_sdr_white_nit,
        display_hdr_white_nit=display_hdr_white_nit)
    hdr_img_non_linear = non_linearize_output_image(
        img_linear=hdr_img_linear, tf_name=hdr_tf_name, cs_name=hdr_cs_name)
    fname = create_out_hdr_fname_one_file(
        hdr_fname=hdr_fname,
        display_sdr_white_nit=display_sdr_white_nit,
        display_hdr_white_nit=display_hdr_white_nit)
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, hdr_img_non_linear)


def debug_new_w_method_with_random_img():
    sdr_fname = "./img/SDR_Random_P3D65.png"
    hdr_fname = "./img/HDR_Random_P3D65.png"
    sdr_cs_name = cs.P3_D65
    hdr_cs_name = cs.P3_D65
    sdr_tf_name = tf.GAMMA24
    hdr_tf_name = tf.ST2084

    create_and_save_gain_map(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname,
        sdr_cs_name=sdr_cs_name, hdr_cs_name=hdr_cs_name,
        sdr_tf_name=sdr_tf_name, hdr_tf_name=hdr_tf_name)

    display_sdr_white_nit = 100
    display_hdr_white_nit = 10000
    gain_map_img_fname = create_gain_map_fname(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname)
    hdr_img_linear = apply_gain_map(
        sdr_fname=sdr_fname, gain_map_img_fname=gain_map_img_fname,
        sdr_cs_name=sdr_cs_name, sdr_tf_name=sdr_tf_name,
        display_sdr_white_nit=display_sdr_white_nit,
        display_hdr_white_nit=display_hdr_white_nit)
    hdr_img_non_linear = non_linearize_output_image(
        img_linear=hdr_img_linear, tf_name=hdr_tf_name, cs_name=hdr_cs_name)
    fname = create_out_hdr_fname_one_file(
        hdr_fname=hdr_fname,
        display_sdr_white_nit=display_sdr_white_nit,
        display_hdr_white_nit=display_hdr_white_nit)
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, hdr_img_non_linear)


def draw_luminance_info(
        img, weight, src_sdr_white, src_hdr_white,
        display_sdr_white, display_hdr_white,
        min_hdr_capacity, max_hdr_capacity):
    # create instance
    font_color = (0.5, 0.5, 0.5)
    stroke_color = (0.0, 0.0, 0.0)
    # text = f"S_SDR_W: {src_sdr_white:4} nits,  "
    # text += f"S_HDR_W: {int(round(src_hdr_white)):5} nits\n"
    text = f"SDR_white: {display_sdr_white:3} nits,  "
    text += f"HDR_white: {display_hdr_white:5} nits\n"
    hdr_capacity = np.log2(display_hdr_white/display_sdr_white)
    text += f"Current HDR_Capacity: H = {hdr_capacity:.3f}\n"
    text += f"WEIGHT: W = {weight:.3f}"
    pos = [60, 30]
    text_draw_ctrl = fc2.TextDrawControl(
        text=text, font_color=font_color,
        font_size=42, font_path=fc2.NOTO_SANS_MONO_BOLD,
        stroke_width=4, stroke_fill=stroke_color)

    text_draw_ctrl.draw(img=img, pos=pos)


def thread_wrapper_apply_gain_map_seq(args):
    apply_gain_map_seq(**args)


def apply_gain_map_seq(
        idx, sdr_fname, hdr_fname, output_dir,
        sdr_cs_name, hdr_cs_name, sdr_tf_name, hdr_tf_name,
        src_sdr_white, src_hdr_white,
        display_sdr_white_nit, display_hdr_white_nit,
        fixed_display_white=True):
    gain_map_img_fname = create_gain_map_fname(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname)
    hdr_img_linear = apply_gain_map(
        sdr_fname=sdr_fname, gain_map_img_fname=gain_map_img_fname,
        sdr_cs_name=sdr_cs_name, sdr_tf_name=sdr_tf_name,
        display_sdr_white_nit=display_sdr_white_nit,
        display_hdr_white_nit=display_hdr_white_nit)

    # for debug
    metadata_fname = create_gain_map_metadata_fname(gain_map_img_fname)
    _, _, min_hdr_capacity, max_hdr_capacity\
        = load_gain_map_meatadata(metadata_fname=metadata_fname)
    weight = calc_weight_w(
        hdr_white=display_hdr_white_nit, sdr_white=display_sdr_white_nit,
        min_val=min_hdr_capacity, max_val=max_hdr_capacity)
    draw_luminance_info(
        img=hdr_img_linear, weight=weight,
        src_sdr_white=src_sdr_white, src_hdr_white=src_hdr_white,
        display_sdr_white=display_sdr_white_nit,
        display_hdr_white=display_hdr_white_nit,
        min_hdr_capacity=min_hdr_capacity, max_hdr_capacity=max_hdr_capacity)

    hdr_img_non_linear = non_linearize_output_image(
        img_linear=hdr_img_linear, tf_name=hdr_tf_name, cs_name=hdr_cs_name)
    if fixed_display_white:
        prefix = f"Mhi_{max_hdr_capacity:.2f}_REF_WHITE_{display_sdr_white_nit}"
    else:
        prefix = f"PEAK_WHITE_{display_hdr_white_nit}"
    fname = create_out_seq_hdr_fname(
        sdr_fname=sdr_fname, idx=idx, base_dir=output_dir,
        prefix=prefix)
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, hdr_img_non_linear)


def calc_peak_luminance_from_st2084_img(fname):
    """
    Note
    ----
    The return value is not Y value.
    The return value is calulurated from each color value.
    """
    img = tpg.img_read_as_float(fname)
    linear = tf.eotf_to_luminance(img, tf.ST2084)
    peak_luminance = np.max(linear)

    return peak_luminance


def debug_check_effect_of_weight_w(
    sdr_fname="./img/SDR_TyTP_P3D65.png",
    hdr_fname="./img/HDR_tyTP_P3D65.png",
    src_sdr_white=203,
    src_hdr_white=None,
    display_sdr_white_nit=203,
    num_of_sample=8
):
    sdr_cs_name = cs.P3_D65
    hdr_cs_name = cs.P3_D65
    sdr_tf_name = tf.GAMMA24
    hdr_tf_name = tf.ST2084
    create_and_save_gain_map(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname,
        sdr_white=src_sdr_white,
        sdr_cs_name=sdr_cs_name, hdr_cs_name=hdr_cs_name,
        sdr_tf_name=sdr_tf_name, hdr_tf_name=hdr_tf_name)

    if src_hdr_white is None:
        src_hdr_white = calc_peak_luminance_from_st2084_img(
            fname=hdr_fname)

    st_log = np.log2(display_sdr_white_nit)
    ed_log = np.log2(10000)
    log_scale = np.linspace(st_log, ed_log, num_of_sample)
    display_hdr_white_nit_list = 2 ** log_scale

    display_hdr_white_nit_list\
        = np.round(display_hdr_white_nit_list).astype(np.uint16)

    total_process_num = len(display_hdr_white_nit_list)
    block_process_num = int(cpu_count() / 2 + 0.999)
    block_num = int(round(total_process_num / block_process_num + 0.5))
    output_dir = "/work/overuse/2023/10_Gain_Map_HDR/"

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
                src_sdr_white=src_sdr_white, src_hdr_white=src_hdr_white,
                display_sdr_white_nit=display_sdr_white_nit,
                display_hdr_white_nit=display_hdr_white_nit)
            # apply_gain_map_seq(**d)
            args.append(d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_apply_gain_map_seq, args)


def create_hdr_img_seq_fix_display_white_vary_display_peak(
    sdr_fname="./img/SDR_TyTP_P3D65.png",
    hdr_fname="./img/HDR_tyTP_P3D65.png",
    src_sdr_white=203,
    src_hdr_white=None,
    min_hdr_capacity=np.log2(203/HDR_REF_WHITE),
    max_hdr_capacity=np.log2(1000/HDR_REF_WHITE),
    display_sdr_white_nit=203,
    num_of_sample=8
):
    sdr_cs_name = cs.P3_D65
    hdr_cs_name = cs.P3_D65
    sdr_tf_name = tf.GAMMA24
    hdr_tf_name = tf.ST2084
    create_and_save_gain_map(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname,
        sdr_white=src_sdr_white,
        sdr_cs_name=sdr_cs_name, hdr_cs_name=hdr_cs_name,
        sdr_tf_name=sdr_tf_name, hdr_tf_name=hdr_tf_name,
        min_hdr_capacity=min_hdr_capacity, max_hdr_capacity=max_hdr_capacity)

    if src_hdr_white is None:
        src_hdr_white = calc_peak_luminance_from_st2084_img(
            fname=hdr_fname)

    st_log = np.log2(display_sdr_white_nit)
    ed_log = np.log2(10000)
    log_scale = np.linspace(st_log, ed_log, num_of_sample)
    display_hdr_white_nit_list = 2 ** log_scale

    display_hdr_white_nit_list\
        = np.round(display_hdr_white_nit_list).astype(np.uint16)

    total_process_num = len(display_hdr_white_nit_list)
    block_process_num = int(cpu_count() / 3 + 0.999)
    block_num = int(round(total_process_num / block_process_num + 0.5))
    output_dir = "/work/overuse/2023/10_Gain_Map_HDR/"

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
                src_sdr_white=src_sdr_white, src_hdr_white=src_hdr_white,
                display_sdr_white_nit=display_sdr_white_nit,
                display_hdr_white_nit=display_hdr_white_nit,
                fixed_display_white=True)
            # apply_gain_map_seq(**d)
            args.append(d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_apply_gain_map_seq, args)


def create_hdr_img_seq_fix_display_peak_vary_display_white(
    sdr_fname="./img/SDR_TyTP_P3D65.png",
    hdr_fname="./img/HDR_tyTP_P3D65.png",
    src_sdr_white=203,
    src_hdr_white=None,
    min_hdr_capacity=np.log2(203/HDR_REF_WHITE),
    max_hdr_capacity=np.log2(1000/HDR_REF_WHITE),
    display_peak_nit=1000,
    num_of_sample=8
):
    sdr_cs_name = cs.P3_D65
    hdr_cs_name = cs.P3_D65
    sdr_tf_name = tf.GAMMA24
    hdr_tf_name = tf.ST2084
    create_and_save_gain_map(
        sdr_fname=sdr_fname, hdr_fname=hdr_fname,
        sdr_white=src_sdr_white,
        sdr_cs_name=sdr_cs_name, hdr_cs_name=hdr_cs_name,
        sdr_tf_name=sdr_tf_name, hdr_tf_name=hdr_tf_name,
        min_hdr_capacity=min_hdr_capacity, max_hdr_capacity=max_hdr_capacity)

    if src_hdr_white is None:
        src_hdr_white = calc_peak_luminance_from_st2084_img(
            fname=hdr_fname)

    st_log = np.log2(10)
    ed_log = np.log2(display_peak_nit)
    log_scale = np.linspace(st_log, ed_log, num_of_sample)
    display_sdr_white_nit_list = 2 ** log_scale

    display_sdr_white_nit_list\
        = np.round(display_sdr_white_nit_list).astype(np.uint16)

    total_process_num = len(display_sdr_white_nit_list)
    block_process_num = int(cpu_count() / 2 + 0.999)
    block_num = int(round(total_process_num / block_process_num + 0.5))
    output_dir = "/work/overuse/2023/10_Gain_Map_HDR/"

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            display_sdr_white_nit = display_sdr_white_nit_list[l_idx]
            d = dict(
                idx=l_idx, sdr_fname=sdr_fname, hdr_fname=hdr_fname,
                output_dir=output_dir,
                sdr_cs_name=sdr_cs_name, hdr_cs_name=hdr_cs_name,
                sdr_tf_name=sdr_tf_name, hdr_tf_name=hdr_tf_name,
                src_sdr_white=src_sdr_white, src_hdr_white=src_hdr_white,
                display_sdr_white_nit=display_sdr_white_nit,
                display_hdr_white_nit=display_peak_nit,
                fixed_display_white=False)
            # apply_gain_map_seq(**d)
            args.append(d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_apply_gain_map_seq, args)


def calc_diff_two_images_hdr_sdr(
        hdr_dst_fname="./debug/SDR_Random_P3D65.png_100_100_ty_w.png",
        sdr_src_fname="./img/SDR_Random_P3D65.png"):
    img1 = tpg.img_read(hdr_dst_fname)
    img2 = tpg.img_read_as_float(sdr_src_fname)
    img2_linear = tf.eotf_to_luminance(img2, tf.GAMMA24)
    img2 = tf.oetf_from_luminance(img2_linear, tf.ST2084)
    img2 = np.round(img2 * 0xFFFF).astype(np.uint16)
    print(np.min(img2))
    diff = img1.astype(np.int32) - img2.astype(np.int32)
    print(f"min={np.min(diff)}")
    print(f"max={np.max(diff)}")
    print(f"ave={np.average(diff)}")
    print(f"std={np.std(diff)}")


def calc_diff_two_images_hdr_hdr(
        hdr_dst_fname="./debug/SDR_Random_P3D65.png_100_100_ty_w.png",
        hdr_src_fname="./img/HDR_Random_P3D65.png"):
    img1 = tpg.img_read(hdr_dst_fname)
    img2 = tpg.img_read(hdr_src_fname)
    print(np.min(img2))
    diff = img1.astype(np.int32) - img2.astype(np.int32)
    print(f"min={np.min(diff)}")
    print(f"max={np.max(diff)}")
    print(f"ave={np.average(diff)}")
    print(f"std={np.std(diff)}")


def create_random_img():
    width = 1920
    height = 1080
    num_of_color = 3
    max_10bit = 1023
    max_10bit_plus_one = 1024
    np.random.seed(101)
    sdr_img = np.random.randint(
        0, max_10bit_plus_one, (height, width, num_of_color)) / max_10bit
    hdr_img = np.random.randint(
        0, max_10bit_plus_one, (height, width, num_of_color)) / max_10bit
    sdr_fname = "./img/SDR_Random_P3D65.png"
    hdr_fname = "./img/HDR_Random_P3D65.png"
    print(sdr_fname)
    tpg.img_wirte_float_as_16bit_int(sdr_fname, sdr_img)
    print(hdr_fname)
    tpg.img_wirte_float_as_16bit_int(hdr_fname, hdr_img)


def debug_dpx_to_png(src, dst):
    img = read_image(src)
    tpg.img_wirte_float_as_16bit_int(dst, img)


def debug_create_sub_img(hdr_fname, sdr_fname):
    hdr_img = tpg.img_read_as_float(hdr_fname)
    sdr_img = tpg.img_read_as_float(sdr_fname)

    hdr_linear = tf.eotf_to_luminance(hdr_img, tf.ST2084)
    sdr_linear = tf.eotf_to_luminance(sdr_img, tf.GAMMA24)
    sdr_linear = sdr_linear * HDR_REF_WHITE / tf.REF_WHITE_LUMINANCE

    hdr_sdr_linear = np.clip(hdr_linear - sdr_linear, 0.0, 10000.0)
    sdr_hdr_linear = np.clip(sdr_linear - hdr_linear, 0.0, 10000.0)

    hdr_sdr_img = tf.oetf_from_luminance(hdr_sdr_linear, tf.ST2084)
    sdr_hdr_img = tf.oetf_from_luminance(sdr_hdr_linear, tf.ST2084)

    tpg.img_wirte_float_as_16bit_int("./debug/hdr-sdr.png", hdr_sdr_img)
    tpg.img_wirte_float_as_16bit_int("./debug/sdr-hdr.png", sdr_hdr_img)


def debug_create_sdr_img_using_tonemapping(hdr_fname):
    sdr_fname = hdr_fname.replace("HDR", "SDR")
    hdr_img = tpg.img_read_as_float(hdr_fname)
    tm_func = create_ty_tonemap_v2_func()
    sdr_img_2084 = tm_func(hdr_img)
    sdr_img_liner = tf.eotf_to_luminance(sdr_img_2084, tf.ST2084)
    sdr_img_liner = sdr_img_liner / HDR_REF_WHITE * tf.REF_WHITE_LUMINANCE
    sdr_img_gm24 = tf.oetf_from_luminance(sdr_img_liner, tf.GAMMA24)

    print(sdr_fname)
    tpg.img_wirte_float_as_16bit_int(sdr_fname, sdr_img_gm24)


def debug_func():
    sdr_fname = "./debug/SDR_ohori.png"
    hdr_fname = "./debug/HDR_ohori.png"
    # debug_simple_implementation(
    #     sdr_fname=sdr_fname,
    #     hdr_fname=hdr_fname,
    #     min_hdr_capacity=np.log2(203/HDR_REF_WHITE),
    #     max_hdr_capacity=np.log2(1000/HDR_REF_WHITE),
    #     display_sdr_white_nit=203,
    #     display_hdr_white_nit=400)

    peak_luminance_list = [
        203, 400, 600, 1000, 4000, 10000
    ]
    # hdr_src_clip_luminance_list = [
    #     600, 1000, 4000, 10000
    # ]
    # for peak_luminance in peak_luminance_list:
    #     for hdr_clip_lumi in hdr_src_clip_luminance_list:
    #         sdr_fname = "./img/SDR_TyTP_P3D65.png"
    #         hdr_fname = f"./img/HDR_tyTP_P3D65_{hdr_clip_lumi}.png"
    #         debug_simple_implementation(
    #             sdr_fname=sdr_fname,
    #             hdr_fname=hdr_fname,
    #             min_hdr_capacity=np.log2(203/HDR_REF_WHITE),
    #             max_hdr_capacity=np.log2(1000/HDR_REF_WHITE),
    #             display_sdr_white_nit=203,
    #             display_hdr_white_nit=peak_luminance)

    # for peak_luminance in peak_luminance_list:
    #     debug_simple_implementation(
    #         sdr_fname="./img/SDR_tyTP_P3D65.png",
    #         hdr_fname="./img/HDR_tyTP_P3D65.png",
    #         display_sdr_white_nit=203,
    #         display_hdr_white_nit=peak_luminance)

    # debug_check_effect_of_weight_w(
    #     sdr_fname="./img/SDR_TyTP_P3D65.png",
    #     hdr_fname="./img/HDR_tyTP_P3D65.png",
    #     src_sdr_white=203,
    #     src_hdr_white=10000,
    #     display_sdr_white_nit=203)

    # hdr_src_clip_luminance_list = [
    #     600, 1000, 4000, 10000
    # ]
    # for hdr_clip_lumi in hdr_src_clip_luminance_list:
    #     hdr_fname = f"./img/HDR_tyTP_P3D65_{hdr_clip_lumi}.png"
    #     debug_check_effect_of_weight_w(
    #         sdr_fname="./img/SDR_TyTP_P3D65.png",
    #         hdr_fname=hdr_fname,
    #         src_sdr_white=203,
    #         src_hdr_white=hdr_clip_lumi,
    #         display_sdr_white_nit=203)

    # debug_simple_implementation(
    #     sdr_fname="./debug/HDR_komorebi.png",
    #     hdr_fname="./debug/SDR_komorebi.png",
    #     display_sdr_white_nit=203,
    #     display_hdr_white_nit=10000)

    sdr_hdr_fname_list = [
        # ["./debug/SDR_komorebi_resolve.png", "./debug/HDR_komorebi.png"],
        # ["./debug/SDR_ohori_resolve.png", "./debug/HDR_ohori.png"],
        # ["./debug/SDR_kougen_resolve.png", "./debug/HDR_kougen.png"],
        ["./img/SDR_TyTP_P3D65.png", "./img/HDR_tyTP_P3D65.png"],
        # ["./debug/SDR_komorebi.png", "./debug/HDR_komorebi.png"],
        # ["./debug/SDR_komorebi.png", "./debug/HDR_komorebi_10000_pixel.png"],
        # ["./debug/SDR_ohori.png", "./debug/HDR_ohori.png"],
        # ["./debug/SDR_kougen.png", "./debug/HDR_kougen.png"],
    ]

    for sdr_hdr_fname in sdr_hdr_fname_list:
        sdr_fname = sdr_hdr_fname[0]
        hdr_fname = sdr_hdr_fname[1]
        create_hdr_img_seq_fix_display_white_vary_display_peak(
            sdr_fname=sdr_fname,
            hdr_fname=hdr_fname,
            src_sdr_white=HDR_REF_WHITE,
            src_hdr_white=None,
            min_hdr_capacity=np.log2(203/HDR_REF_WHITE),
            max_hdr_capacity=np.log2(10000/HDR_REF_WHITE),
            display_sdr_white_nit=203,
            num_of_sample=180)
        # create_hdr_img_seq_fix_display_peak_vary_display_white(
        #     sdr_fname=sdr_fname,
        #     hdr_fname=hdr_fname,
        #     src_sdr_white=HDR_REF_WHITE,
        #     src_hdr_white=None,
        #     min_hdr_capacity=np.log2(203/HDR_REF_WHITE),
        #     max_hdr_capacity=np.log2(1000/HDR_REF_WHITE),
        #     display_peak_nit=1000,
        #     num_of_sample=10)

    # create_random_img()
    # debug_new_w_method_with_random_img()
    # debug_dpx_to_png(
    #     src="./img/Sparks_01_38_28_SDR_P3D65.dpx",
    #     dst="./img/Sparks_01_38_28_SDR_P3D65.png"
    # )
    # debug_dpx_to_png(
    #     src="./img/Sparks_01_38_28_HDR_P3D65.dpx",
    #     dst="./img/Sparks_01_38_28_HDR_P3D65.png"
    # )
    # debug_check_effect_of_weight_w()

    # debug_check_effect_of_weight_w()

    # debug_create_sub_img(
    #     hdr_fname="./debug/HDR_komorebi.png",
    #     sdr_fname="./debug/SDR_komorebi.png"
    # )
    # debug_create_sdr_img_using_tonemapping(hdr_fname="./debug/HDR_komorebi.png")
    # debug_create_sdr_img_using_tonemapping(hdr_fname="./debug/HDR_tyTP_P3D65.png")


def plot_weighting_parameter_w():
    sdr_white = 203
    x = np.linspace(sdr_white, 10000, 1024)
    capacity = np.log2(x / sdr_white)

    min_capacity_1 = np.log2(203 / sdr_white)
    max_capacity_1 = np.log2(1000 / sdr_white)
    min_capacity_2 = np.log2(203 / sdr_white)
    max_capacity_2 = np.log2(10000 / sdr_white)
    # min_capacity_1 = 1
    # max_capacity_1 = 4
    # min_capacity_2 = 0
    # max_capacity_2 = 5

    w1 = (capacity - min_capacity_1) / (max_capacity_1 - min_capacity_1)
    w1 = np.clip(w1, 0.0, 1.0)
    w2 = (capacity - min_capacity_2) / (max_capacity_2 - min_capacity_2)
    w2 = np.clip(w2, 0.0, 1.0)

    title = "Relationship between $H$ and $W$"

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 7),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=title,
        graph_title_size=None,
        xlabel="HDR Capacity $H$",
        ylabel='Weight parameter $W$',
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
    label1 = f"$M_{{lo}}$={min_capacity_1:.2f}, $M_{{hi}}$={max_capacity_1:.2f}"
    label2 = f"$M_{{lo}}$={min_capacity_2:.2f}, $M_{{hi}}$={max_capacity_2:.2f}"
    ax1.plot(capacity, w1, label=label1)
    ax1.plot(capacity, w2, label=label2)
    pu.show_and_save(
        fig=fig, legend_loc='lower right',
        save_fname="./img/blog_img/w_and_hdr_capcacity.png",
        show=True)


def create_graph_for_blog():
    plot_weighting_parameter_w()


def add_10000nits_pixel():
    src_img_file = "./debug/HDR_komorebi.png"
    dst_img_file = "./debug/HDR_komorebi_10000_pixel.png"

    img = tpg.img_read_as_float(src_img_file)
    img[321, 1027] = np.array([1.0, 1.0, 1.0])

    tpg.img_wirte_float_as_16bit_int(dst_img_file, img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
    # create_graph_for_blog()

    # sdr_white = 203
    # hdr_white = 600
    # hdr_capacity = np.log2(hdr_white/sdr_white)
    # print(hdr_capacity)
    # sdr_white = 80
    # hdr_white = 600
    # hdr_capacity = np.log2(hdr_white/sdr_white)
    # print(hdr_capacity)
    # sdr_white = 400
    # hdr_white = 600
    # hdr_capacity = np.log2(hdr_white/sdr_white)
    # print(hdr_capacity)
