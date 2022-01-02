# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import subprocess

from test_pattern_generator2 import img_read
from color_space import D65

# import third-party libraries
import numpy as np
from colour import write_image, xy_to_XYZ
from colour.models import RGB_COLOURSPACES, RGB_COLOURSPACE_ACESCG,\
    RGB_COLOURSPACE_BT709
from colour.adaptation import matrix_chromatic_adaptation_VonKries
from scipy import linalg

# import my libraries
import test_pattern_generator2 as tpg
from davinci17_cms_explore import get_media_src_fname_sdr,\
    get_media_src_fname_hdr, get_media_src_fname_exr,\
    EXR_MIN_EXPOSURE, EXR_MAX_EXPOSURE
import font_control as fc
import transfer_functions as tf
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


# FONT_PATH = "../../font/NotoSansMono-Bold.ttf"
FONT_PATH = "../../font/NotoSansMono-Medium.ttf"


# d60_to_d65_mtx = np.array(
#     [[9.15403080e-01, -7.30439169e-03, 1.55226464e-02],
#      [-5.27365930e-03, 9.25092018e-01, 6.18063317e-03],
#      [3.27220257e-04, 1.04119480e-03, 9.98631585e-01]]
# )
d60_to_d65_mtx = np.array(
    [[0.9363736723572331, 0.0656966352314790, -0.0020703075887115],
     [-0.0025461715321907, 1.0026784278722174, -0.0001322563400272],
     [0.0006046772188210, 0.0095422518235398, 0.9898530709576396]]
)
d65_to_d60_mtx = linalg.inv(d60_to_d65_mtx)
print(d65_to_d60_mtx)


def apply_matrix(src, mtx):
    """
    src: (N, M, 3)
    mtx: (3, 3)
    """
    shape_bak = src.shape
    a = src[..., 0]*mtx[0][0] + src[..., 1]*mtx[0][1] + src[..., 2]*mtx[0][2]
    b = src[..., 0]*mtx[1][0] + src[..., 1]*mtx[1][1] + src[..., 2]*mtx[1][2]
    c = src[..., 0]*mtx[2][0] + src[..., 1]*mtx[2][1] + src[..., 2]*mtx[2][2]

    return np.dstack([a, b, c]).reshape(shape_bak)


def create_src_png_img_seq_sdr():
    fps = 24
    sec = 2/24
    width = 1920
    height = 1080
    line = np.linspace(0, 1, width)
    img = tpg.h_mono_line_to_img(line, height)
    color_space_name = cs.BT709
    white_point = cs.D65

    # add color checker on 2nd line
    color_checker_rgb_linear = tpg.generate_color_checker_rgb_value(
        color_space=RGB_COLOURSPACES[color_space_name],
        target_white=white_point)
    color_checker_rgb = tf.oetf(color_checker_rgb_linear, tf.GAMMA24)
    img[1, 0:len(color_checker_rgb)] = color_checker_rgb

    # add color checker
    cc_st_pos = [int(width * 0.05), int(width * 0.1)]
    cc_width = int(width * 0.4)
    color_checker_linear = tpg.make_color_checker_image(
        color_space_name=color_space_name, target_white=white_point,
        width=cc_width, padding_rate=0.01)
    color_checker_gm24 = tf.oetf(color_checker_linear, tf.GAMMA24)
    tpg.merge(img, color_checker_gm24, pos=cc_st_pos)

    frame = int(fps * sec)

    for idx in range(frame):
        fname = get_media_src_fname_sdr(idx=idx)
        print(fname)
        tpg.img_wirte_float_as_16bit_int(fname, img)


def create_src_png_img_seq_hdr():
    fps = 24
    sec = 2/24
    width = 1920
    height = 1080
    line = np.linspace(0, 1, width)
    img = tpg.h_mono_line_to_img(line, height)
    color_space_name = cs.BT2020
    white_point = cs.D65

    # add color checker on 2nd line
    color_checker_rgb_luminance = tpg.generate_color_checker_rgb_value(
        color_space=RGB_COLOURSPACES[color_space_name],
        target_white=white_point) * 100
    color_checker_rgb = tf.oetf_from_luminance(
        color_checker_rgb_luminance, tf.ST2084)
    img[1, 0:len(color_checker_rgb)] = color_checker_rgb

    # add color checker
    cc_st_pos = [int(width * 0.05), int(width * 0.1)]
    cc_width = int(width * 0.4)
    color_checker_linear_luminance = tpg.make_color_checker_image(
        color_space_name=color_space_name, target_white=white_point,
        width=cc_width, padding_rate=0.01) * 100
    color_checker_st2084 = tf.oetf_from_luminance(
        color_checker_linear_luminance, tf.ST2084)
    tpg.merge(img, color_checker_st2084, pos=cc_st_pos)

    frame = int(fps * sec)

    for idx in range(frame):
        fname = get_media_src_fname_hdr(idx=idx)
        print(fname)
        tpg.img_wirte_float_as_16bit_int(fname, img)


def make_src_video_sdr():
    in_fname = str(TP_SRC_PATH / "src_png_0000.png")
    in_fname_ffmpeg = in_fname.replace("0000", r"%4d")
    out_fname = get_media_src_fname_sdr()
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt709',# '-color_trc', 'auto',
        '-colorspace', 'bt709',
        '-r', '24', '-i', in_fname_ffmpeg,
        '-an',
        '-c:v', 'hevc',
        '-movflags', 'write_colr',
        '-pix_fmt', 'yuv444p12le', '-qp', '0',
        '-color_primaries', 'bt709',# '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        str(out_fname), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def make_src_video_hdr():
    in_fname = str(TP_SRC_PATH / "src_png_0000.png")
    in_fname_ffmpeg = in_fname.replace("0000", r"%4d")
    out_fname = get_media_src_fname_hdr()
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-r', '24', '-i', in_fname_ffmpeg,
        '-an',
        '-c:v', 'hevc',
        '-movflags', 'write_colr',
        '-pix_fmt', 'yuv444p12le', '-qp', '0',
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        str(out_fname), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def draw_scale(img, mid_gray, min_exposure, max_exposure):
    # set parameters
    width = img.shape[1]
    height = img.shape[0]
    font_color = np.array([0.5, 0.0, 0.5])
    scale_color = tf.eotf(font_color, tf.SRGB)
    font_size = 20
    major_v_len = int(height * 0.015)
    minor_v_len = major_v_len // 2

    # calc coordinate
    major_pos = np.int16(np.linspace(0, width, max_exposure-min_exposure+1))
    minor_length = major_pos[1] - major_pos[0]
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    minor_offset = np.uint16(np.round(np.log10(x) * minor_length))

    # plor major and minor scale
    for pos_h in major_pos[:-1]:
        st_pos_v = height - major_v_len
        ed_pos_v = height
        st_pos_h = pos_h
        ed_pos_h = st_pos_h + 2
        img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = scale_color
        for offset_h in minor_offset:
            st_pos_v = height - minor_v_len
            ed_pos_v = height
            st_pos_h = pos_h + offset_h
            ed_pos_h = st_pos_h + 2
            img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = scale_color

    # draw text
    for idx, pos_h in enumerate(major_pos):
        if (idx == 0) or (idx == len(major_pos) - 1):
            continue
        exposure = min_exposure + idx
        text = f"1.0e{exposure}"
        text_width, text_height = fc.get_text_width_height(
            text, FONT_PATH, font_size)
        pos = (
            pos_h - text_width // 2,
            height - major_v_len - int(text_height * 1.4))
        text_drawer = fc.TextDrawer(
            img, text=text, pos=pos,
            font_color=font_color, font_size=font_size,
            font_path=FONT_PATH,
            bg_transfer_functions=tf.LINEAR,
            fg_transfer_functions=tf.SRGB)
        text_drawer.draw()


def draw_scale_for_18_gray(img, mid_gray, min_exposure, max_exposure):
    # set parameters
    width = img.shape[1]
    height = img.shape[0]
    font_color = np.array([0.5, 0.0, 0.5])
    scale_color = tf.eotf(font_color, tf.SRGB)
    font_size = 16
    major_v_len = int(height * 0.015)

    # calc coordinate
    major_pos = np.int16(np.linspace(0, width, max_exposure-min_exposure+1))

    # plor major and minor scale
    for pos_h in major_pos[:-1]:
        st_pos_v = height - major_v_len
        ed_pos_v = height
        st_pos_h = pos_h
        ed_pos_h = st_pos_h + 2
        img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = scale_color

    # draw text
    for idx, pos_h in enumerate(major_pos):
        if (idx == 0) or (idx == len(major_pos) - 1):
            continue
        exposure = min_exposure + idx
        text = f"0.18*\n2^{exposure}"
        text_width, text_height = fc.get_text_width_height(
            text, FONT_PATH, font_size)
        pos = (
            pos_h - text_width // 2,
            height - major_v_len - int(text_height * 1.4))
        text_drawer = fc.TextDrawer(
            img, text=text, pos=pos,
            font_color=font_color, font_size=font_size,
            font_path=FONT_PATH,
            bg_transfer_functions=tf.LINEAR,
            fg_transfer_functions=tf.SRGB)
        text_drawer.draw()


def calc_chromatic_adaptation_input_rgb():
    # m1 is ACES_AP1 to XYZ
    m1_inv = RGB_COLOURSPACE_ACESCG.matrix_XYZ_to_RGB
    identity_mtx = np.identity(3)
    rgb = m1_inv.dot(identity_mtx).T

    return rgb


def create_src_exr_img_seq():
    fps = 24
    sec = 2/24
    width = 1920
    height = 1080
    mid_gray = 1.0
    min_exposure = EXR_MIN_EXPOSURE
    max_exposure = EXR_MAX_EXPOSURE
    x = np.linspace(0, 1, width)
    line = tpg.shaper_func_log10_to_linear(
        x, mid_gray=mid_gray,
        min_exposure=min_exposure, max_exposure=max_exposure)
    img = tpg.h_mono_line_to_img(line, height)
    color_space_name = cs.ACES_AP1
    white_point = D65
    ca_mtx = d65_to_d60_mtx

    # add color checker on 2nd line
    color_checker_rgb_linear_d65 = tpg.generate_color_checker_rgb_value(
        color_space=RGB_COLOURSPACES[color_space_name],
        target_white=white_point)
    color_checker_rgb_linear_d60 = apply_matrix(
        src=color_checker_rgb_linear_d65, mtx=ca_mtx)
    img[1, 0:len(color_checker_rgb_linear_d60)] = color_checker_rgb_linear_d60

    # add chromatic adaptation matrix checker
    chromatic_adaptation_input_rgb = calc_chromatic_adaptation_input_rgb()
    img[3, 0:len(chromatic_adaptation_input_rgb)]\
        = chromatic_adaptation_input_rgb

    # add color checker
    cc_st_pos = [int(width * 0.05), int(width * 0.1)]
    cc_width = int(width * 0.4)
    color_checker_rgb_linear_d65 = tpg.make_color_checker_image(
        color_space_name=color_space_name, target_white=white_point,
        width=cc_width, padding_rate=0.01)
    color_checker_rgb_linear_d60 = apply_matrix(
        src=color_checker_rgb_linear_d65, mtx=ca_mtx)
    tpg.merge(img, color_checker_rgb_linear_d60, pos=cc_st_pos)

    draw_scale(
        img=img, mid_gray=mid_gray, min_exposure=min_exposure,
        max_exposure=max_exposure)

    frame = int(fps * sec)

    for idx in range(frame):
        fname = get_media_src_fname_exr(idx=idx)
        print(fname)
        write_image(
            image=img, path=fname, bit_depth='float32', method="OpenImageIO")


def create_exr_img_tp_18_gray_base():
    width = 1920
    height = 1080
    mid_gray = 0.18
    min_exposure = -16
    max_exposure = 12
    x = np.linspace(0, 1, width)
    line = tpg.shaper_func_log2_to_linear(
        x, mid_gray=mid_gray,
        min_exposure=min_exposure, max_exposure=max_exposure)
    img = tpg.h_mono_line_to_img(line, height)

    draw_scale_for_18_gray(
        img=img, mid_gray=mid_gray, min_exposure=min_exposure,
        max_exposure=max_exposure)

    fname = f"./img/exr_tp_18_gray_{min_exposure}_to_{max_exposure}.exr"
    print(fname)
    write_image(
        image=img, path=fname, bit_depth='float32', method="OpenImageIO")


def get_d60_to_d65_chromatic_adaptation_matrix():
    fname = "./img/d60_to_d65_00086400.exr"
    img = img_read(fname)
    dst_r = img[3, 0, :]
    dst_g = img[3, 1, :]
    dst_b = img[3, 2, :]
    print(dst_r)
    print(dst_g)
    print(dst_b)
    m2_mtx = RGB_COLOURSPACE_BT709.matrix_RGB_to_XYZ
    m11_m21_m31 = m2_mtx.dot(dst_r)
    m12_m22_m32 = m2_mtx.dot(dst_g)
    m13_m23_m33 = m2_mtx.dot(dst_b)
    print(m11_m21_m31)
    print(m12_m22_m32)
    print(m13_m23_m33)

    mtx = np.array([m11_m21_m31, m12_m22_m32, m13_m23_m33]).T
    print("before normalize")
    print(mtx)
    normalize_num = np.sum(mtx, axis=-1)[1]
    print(f"nnn = {normalize_num, np.sum(mtx, axis=-1)}")
    mtx = mtx / normalize_num
    print(f"DaVinci D60 to D65 chromatic adaptation matrix = \n{mtx}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_src_png_img_seq_sdr()
    # create_src_png_img_seq_hdr()
    create_src_exr_img_seq()

    # create_exr_img_tp_18_gray_base()
    # calc_chromatic_adaptation_input_rgb()
    # get_d60_to_d65_chromatic_adaptation_matrix()
