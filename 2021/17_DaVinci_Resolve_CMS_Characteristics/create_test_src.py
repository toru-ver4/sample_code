# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import subprocess

# import third-party libraries
import numpy as np
from colour import write_image

# import my libraries
import test_pattern_generator2 as tpg
from davinci17_cms_explore import get_media_src_fname_sdr,\
    get_media_src_fname_hdr, get_media_src_fname_exr,\
    EXR_MIN_EXPOSURE, EXR_MAX_EXPOSURE
import font_control as fc
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


# FONT_PATH = "../../font/NotoSansMono-Bold.ttf"
FONT_PATH = "../../font/NotoSansMono-Medium.ttf"


def create_src_png_img_seq():
    fps = 24
    sec = 1
    width = 1920
    height = 1080
    line = np.linspace(0, 1, width)
    img = tpg.h_mono_line_to_img(line, height)

    frame = fps * sec

    for idx in range(frame):
        fname = get_media_src_fname_sdr(idx=idx)
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


def create_src_exr_img_seq():
    fps = 24
    sec = 1
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

    draw_scale(
        img=img, mid_gray=mid_gray, min_exposure=min_exposure,
        max_exposure=max_exposure)

    frame = fps * sec

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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_src_png_img_seq()
    # create_src_exr_img_seq()
    create_exr_img_tp_18_gray_base()
