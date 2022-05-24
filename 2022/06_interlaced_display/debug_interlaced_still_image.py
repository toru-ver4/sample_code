# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
from asyncore import write
import os
import subprocess

# import third-party libraries
import numpy as np
from colour.io import write_image, read_image
import cv2

# import my libraries
import test_pattern_generator2 as tpg
import font_control2 as fc2
from ty_utility import add_suffix_to_filename

NOTO_FONT_PATH = "C:/Users/toruv/OneDrive/work/sample_code/font/NotoSansMono-Medium.ttf"

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_interlace_mask_img(
        width=640, height=480, black_line_height=2, parity='even'):
    img = np.ones((height, width, 3))
    mask_idx = np.arange(height) % (black_line_height * 2)
    if parity == 'even':
        mask_idx = mask_idx < black_line_height
    elif parity == 'odd':
        mask_idx = mask_idx >= black_line_height
    else:
        raise ValueError('parameter "parity" is invalid')

    img[mask_idx] = 0

    return img


def create_interlaced_still(
        width=1920, height=1080, color_mask=[1, 0, 0], black_line_height=2):
    block_width = width // 2
    plane_img = np.ones((height, block_width, 3)) * np.array(color_mask)
    mask_img = create_interlace_mask_img(
        width=block_width, height=height, black_line_height=black_line_height,
        parity='even')
    interlaced_img = plane_img * mask_img
    plane_img_liner = plane_img ** 2.4
    plane_img_half = (plane_img_liner * 0.5) ** (1/2.4)
    img = np.hstack([interlaced_img, plane_img_half])
    color_str = f"color-{color_mask[0]}r_{color_mask[1]}g_{color_mask[2]}b"
    fname = f"./img/i_still_{width}x{height}_"
    fname += f"l-{black_line_height}px_{color_str}.png"
    print(fname)
    write_image(img, fname, 'uint8')


def interlaced_still_for_debug_420(
        width=1920, height=1080, black_line_height=2):
    plane_img = np.ones((height, width, 3))
    mask_img = create_interlace_mask_img(
        width=width, height=height, black_line_height=black_line_height,
        parity='even')
    inv_mask = 1 - mask_img
    r_img = plane_img * mask_img * np.array([1, 0, 0])
    c_img = plane_img * inv_mask * np.array([0, 1, 1])
    img = r_img + c_img
    fname = f"./img/debug_420_{width}x{height}_"
    fname += f"l-{black_line_height}px.png"
    print(fname)
    write_image(img, fname, 'uint8')


def interlaced_still_for_debug_420_2nd(
        width=1920, height=1080, black_line_height=2):
    plane_img = np.ones((height, width, 3))
    mask_img = create_interlace_mask_img(
        width=width, height=height, black_line_height=black_line_height,
        parity='even')
    inv_mask = 1 - mask_img
    m_img = plane_img * mask_img * np.array([1, 0, 1])
    k_img = plane_img * inv_mask * np.array([0, 0, 0])
    img = m_img + k_img
    fname = f"./img/debug_420_2nd_{width}x{height}_"
    fname += f"l-{black_line_height}px.png"
    print(fname)
    write_image(img, fname, 'uint8')


def analyze_ycbcr420_encode():
    pass


def create_analyze_420_still_fname(offset=[0, 0]):
    fname = f"./img/ana_420_offset_{offset[0]}-{offset[1]}.png"
    return fname


def create_analyze_420_raw_yuv_name(pix_fmt='yuv420p', offset=[0, 0]):
    fname = f"./videos/ana_420_offset_fmt-{pix_fmt}_"
    fname += f"{offset[0]}-{offset[1]}.yuv"
    return fname


def create_analyze_420_x265_out_name(pix_fmt='yuv420p', offset=[0, 0]):
    fname = f"./videos/ana_420_offset_fmt-{pix_fmt}_"
    fname += f"{offset[0]}-{offset[1]}.h265"
    return fname


def create_analyze_420_ffmpe_out_name(pix_fmt='yuv420p', offset=[0, 0]):
    fname = f"./videos/ana_420_offset_fmt-{pix_fmt}_"
    fname += f"{offset[0]}-{offset[1]}.mov"
    return fname


def create_analyze_420_decode_still_name(pix_fmt='yuv420p', offset=[0, 0]):
    fname = f"./img/decode_ana_420_offset_fmt-{pix_fmt}_"
    fname += f"{offset[0]}-{offset[1]}.png"
    return fname


def create_resized_analyze_still_name(pix_fmt='yuv420p', offset=[0, 0]):
    fname = f"./img/resize_ana_420_offset_fmt-{pix_fmt}_"
    fname += f"{offset[0]}-{offset[1]}.png"
    return fname


def create_concat_analyze_src_dst_name(pix_fmt='yuv420p', offset=[0, 0]):
    fname = f"./img/concat_ana_420_offset_fmt-{pix_fmt}_"
    fname += f"{offset[0]}-{offset[1]}.png"
    return fname


def encode_with_ffmpeg(
        pix_fmt='yuv420p', offset=[0, 0]):
    in_fname_ffmpeg = create_analyze_420_still_fname(offset=offset)
    out_fname_ffmpeg = create_analyze_420_ffmpe_out_name(
        pix_fmt=pix_fmt, offset=offset)
    fps = 24
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-r', str(fps), '-loop', '1', '-i', in_fname_ffmpeg, '-t', '5',
        '-c:v', 'hevc',
        # '-chroma_sample_location', 'left',
        '-movflags', 'write_colr',
        '-pix_fmt', pix_fmt, '-x265-params', 'lossless=1',
        '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        str(out_fname_ffmpeg), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def decode_with_ffmpeg(pix_fmt='yuv420p', offset=[0, 0]):
    in_fname_ffmpeg = create_analyze_420_ffmpe_out_name(
        pix_fmt=pix_fmt, offset=offset)
    out_fname_still = create_analyze_420_decode_still_name(
        pix_fmt=pix_fmt, offset=offset)
    cmd = "ffmpeg"
    ops = [
        '-i', in_fname_ffmpeg, '-vframes', '1', out_fname_still, '-y']
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def _crop_and_scaling_core(img):
    crop_px = 32
    mag_rate = 32
    crop_img = img[:crop_px, :crop_px]
    resize_img = cv2.resize(
        crop_img, None, fx=mag_rate, fy=mag_rate,
        interpolation=cv2.INTER_NEAREST)

    return resize_img


def crop_and_scaling_decoded_image(pix_fmt="yuv420p", offset=[0, 0]):
    in_fname = create_analyze_420_decode_still_name(
        pix_fmt=pix_fmt, offset=offset)
    img = read_image(in_fname)
    resize_img = _crop_and_scaling_core(img)
    out_fname = create_resized_analyze_still_name(
        pix_fmt=pix_fmt, offset=offset)
    write_image(resize_img, out_fname, 'uint8')


def add_src_dst_text(img, pix_fmt="yuv420p", offset=[0, 0]):
    font_size = 30
    font_color = [0.5, 0.5, 0.5]
    font_edge_size = int(font_size * 0.2 + 0.5)
    size = img.shape[0]
    src_text = f" Src, pos_offset={[offset[0], offset[1]]}"
    dst_text = f" Encode({pix_fmt}) --> Decode"
    temp_img = img ** 2.4
    font_path = NOTO_FONT_PATH

    # src text
    text_draw_ctrl = fc2.TextDrawControl(
        text=src_text, font_color=font_color,
        font_size=font_size, font_path=font_path,
        stroke_width=font_edge_size, stroke_fill=(0, 0, 0))
    _, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = 0
    pos_v = text_height // 2
    pos = (pos_h, pos_v)
    text_draw_ctrl.draw(img=temp_img, pos=pos)

    # dst text
    text_draw_ctrl = fc2.TextDrawControl(
        text=dst_text, font_color=font_color,
        font_size=font_size, font_path=font_path,
        stroke_width=font_edge_size, stroke_fill=(0, 0, 0))
    _, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = size // 2
    pos_v = text_height // 2
    pos = (pos_h, pos_v)
    text_draw_ctrl.draw(img=temp_img, pos=pos)

    out_img = temp_img ** (1/2.4)

    return out_img


def concat_src_and_dst_ana_img(pix_fmt="yuv420p", offset=[0, 0]):
    src_fname = create_analyze_420_still_fname(offset=offset)
    dst_fname = create_resized_analyze_still_name(
        pix_fmt=pix_fmt, offset=offset)
    src_img = read_image(src_fname)
    dst_img = read_image(dst_fname)
    src_img = _crop_and_scaling_core(src_img)
    size = dst_img.shape[0] // 2
    out_img = np.zeros_like(dst_img)
    out_img[:, :size] = src_img[:, :size]
    out_img[:, size:] = dst_img[:, size:]

    out_img = add_src_dst_text(out_img, pix_fmt=pix_fmt, offset=offset)

    out_fname = create_concat_analyze_src_dst_name(
        pix_fmt=pix_fmt, offset=offset)
    write_image(out_img, out_fname, 'uint8')


def analyze_ycbcr420_encode_each_offset(pix_fmt='yuv420p', offset=[0, 0]):
    """
    Parameters
    ----------
    offset : list(int)
        A list. [x_offset, y_offset]
    """
    # parameters
    width = 1280
    height = 720
    dot_size = [2, 2]

    # create dot pattern
    img = tpg.create_dot_mesh_image(
        width=width, height=height, dot_size=dot_size, st_offset=offset)
    img = img * np.array([1, 0, 1])
    still_fname = create_analyze_420_still_fname(offset=offset)
    write_image(img, still_fname, 'uint8')

    # encode with FFmpeg
    encode_with_ffmpeg(pix_fmt=pix_fmt, offset=offset)

    # # decode with FFmpeg
    decode_with_ffmpeg(pix_fmt=pix_fmt, offset=offset)

    # # crop and scaling (Nearest Neighbor)
    crop_and_scaling_decoded_image(pix_fmt=pix_fmt, offset=offset)

    # # concat src and dst
    concat_src_and_dst_ana_img(pix_fmt=pix_fmt, offset=offset)


def debug_encode_to_yuv(pix_fmt='yuv420p', png_name="a", yuv_name='b'):
    fps = 24
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-r', str(fps), '-loop', '1', '-i', png_name, '-t', '1',
        '-c:v', 'rawvideo', '-pix_fmt', pix_fmt,
        '-chroma_sample_location', 'center',
        yuv_name, '-y']
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def debug_encode_with_x265(
        profile='main444-8', yuv_name='b', h265_name='c'):
    cmd = "x265"
    ops = [
        '--input', yuv_name,
        '--input-depth', '8', '--frames', '24',
        '--input-res', '64x64', '--input-csp', '3', '--fps', '24',
        '--profile', profile, '--output', h265_name]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def debug_x265():
    # create dot pattern
    offset = [0, 0]
    img = tpg.create_dot_mesh_image(
        width=16, height=12, dot_size=[2, 2], st_offset=offset)
    img = img * np.array([1, 0, 1])
    img[:, 0] = np.array([1, 1, 1])
    still_fname = create_analyze_420_still_fname(offset=offset)
    write_image(img, still_fname, 'uint8')

    png_name = create_analyze_420_still_fname(offset=offset)
    yuv_name_420 = create_analyze_420_raw_yuv_name(
        pix_fmt='yuv420p', offset=offset)
    yuv_name_444 = create_analyze_420_raw_yuv_name(
        pix_fmt='yuv444p', offset=offset)
    h265_name_420 = create_analyze_420_x265_out_name(
        pix_fmt='yuv420p', offset=offset)
    h265_name_444 = create_analyze_420_x265_out_name(
        pix_fmt='yuv444p', offset=offset)

    debug_encode_to_yuv(
        pix_fmt='yuv420p', png_name=png_name, yuv_name=yuv_name_420)
    debug_encode_to_yuv(
        pix_fmt='yuv444p', png_name=png_name, yuv_name=yuv_name_444)

    # debug_encode_with_x265(
    #     profile='main', yuv_name=yuv_name_444, h265_name=h265_name_420)
    # debug_encode_with_x265(
    #     profile='main444-8', yuv_name=yuv_name_444, h265_name=h265_name_444)


def debug_sample_interlaced_image():
    filename_list = [
        "./img/movie_SDR_1920x1080_30fps_0000.png",
        "./img/movie_SDR_1920x1080_30fps_0001.png",
        "./img/movie_SDR_1920x1080_30fps_0002.png",
        "./img/movie_SDR_1920x1080_30fps_0003.png"]

    for idx, filename in enumerate(filename_list):
        width1 = 320
        height1 = int(width1 / 16 * 9 + 0.5)
        width2 = width1 * 6
        height2 = int(width2 / 16 * 9 + 0.5)
        img = read_image(filename)
        img = cv2.resize(img, (width1, height1))
        bli_idx = np.arange(height1) % 2 == (idx % 2)
        img[bli_idx] = 0
        img = cv2.resize(
            img, (width2, height2), interpolation=cv2.INTER_NEAREST)
        out_fname = add_suffix_to_filename(fname=filename, suffix='with_bli')
        print(out_fname)
        write_image(img, out_fname, 'uint16')


def encode_with_ffmpeg_for_comp_420_444(
        pix_fmt, in_fname_ffmpeg, out_fname_ffmpeg):
    fps = 24
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-r', str(fps), '-loop', '1', '-i', in_fname_ffmpeg, '-t', '5',
        '-c:v', 'hevc',
        # '-chroma_sample_location', 'left',
        '-movflags', 'write_colr',
        '-pix_fmt', pix_fmt, '-x265-params', 'lossless=1',
        '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        str(out_fname_ffmpeg), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def decode_with_ffmpeg_for_comp_420_444(
        in_fname_ffmpeg, out_fname_still):
    cmd = "ffmpeg"
    ops = [
        '-i', in_fname_ffmpeg, '-vframes', '1', out_fname_still, '-y']
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def comparison_420_444():
    src_png = "./img/movie_SDR_1920x1080_30fps_0000with_bli.png"
    out_420_mov = "./videos/comparison_420_444_420.mov"
    out_444_mov = "./videos/comparison_420_444_444.mov"
    out_420_png = "./img/comparison_420_444_420.png"
    out_444_png = "./img/comparison_420_444_444.png"

    # encode with FFmpeg

    encode_with_ffmpeg_for_comp_420_444(
        pix_fmt='yuv420p', in_fname_ffmpeg=src_png,
        out_fname_ffmpeg=out_420_mov)
    encode_with_ffmpeg_for_comp_420_444(
        pix_fmt='yuv444p', in_fname_ffmpeg=src_png,
        out_fname_ffmpeg=out_444_mov)
    # decode with FFmpeg
    decode_with_ffmpeg_for_comp_420_444(
        in_fname_ffmpeg=out_420_mov,
        out_fname_still=out_420_png)
    decode_with_ffmpeg_for_comp_420_444(
        in_fname_ffmpeg=out_444_mov,
        out_fname_still=out_444_png)


def debug_420_davinci_decorate(
        in_fname, out_fname, text, pos='left'):
    mag_rate = 2
    img = read_image(in_fname)
    height, width = img.shape[:2]
    img = img[:height//mag_rate, :width//mag_rate]
    img = cv2.resize(
        img, None, fx=mag_rate, fy=mag_rate,
        interpolation=cv2.INTER_NEAREST)
    img = img ** 2.4

    font_path = "../../font/NotoSansJP-Medium.otf"

    text_draw_ctrl = fc2.TextDrawControl(
        text=text, font_color=[0.5, 0.5, 0.5],
        font_size=32, font_path=font_path,
        stroke_width=3, stroke_fill=[0, 0, 0])

    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    w_1char = int(text_width / len(text) + 0.5)
    if pos == 'left':
        pos_h = w_1char
    else:
        pos_h = width - text_width - w_1char

    pos_v = text_height // 2
    pos = (pos_h, pos_v)

    text_draw_ctrl.draw(img=img, pos=pos)

    img = img ** (1 / 2.4)

    write_image(img, out_fname, 'uint8')


def encode_mgif_two_images(fname):
    cmd = "ffmpeg"
    ops = [
        '-r', '1', '-i', fname, '-filter_complex',
        "split[a][b];[a]palettegen[pal];[b][pal]paletteuse",
        '-loop', '0', '-r', '2',
        "./img/debug_420_with_davinci.gif", '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def debug_with_davinci_420():
    # interlaced_still_for_debug_420_2nd(
    #     width=640, height=360, black_line_height=2)
    in_fname_420 = "./img/debug_davinci_420.tif"
    out_fname_420 = add_suffix_to_filename(in_fname_420, "_with_text")
    in_fname_444 = "./img/debug_davinci_444.tif"
    out_fname_444 = add_suffix_to_filename(in_fname_444, "_with_text")
    mgif_seq_0 = "./img/debug_davinci_420_00.tif"
    mgif_seq_1 = "./img/debug_davinci_420_01.tif"
    mgif_seq = "./img/debug_davinci_420_%02d.tif"
    print(mgif_seq)

    debug_420_davinci_decorate(
        in_fname=in_fname_420, out_fname=out_fname_420, text=" Main 4:2:0",
        pos='left')
    debug_420_davinci_decorate(
        in_fname=in_fname_444, out_fname=out_fname_444, text=" Main 4:4:4",
        pos='right')

    os.remove(mgif_seq_0)
    os.remove(mgif_seq_1)

    os.rename(out_fname_420, mgif_seq_0)
    os.rename(out_fname_444, mgif_seq_1)

    encode_mgif_two_images(mgif_seq)


def debug_func():
    # img = create_interlace_mask_img(
    #     width=640, height=480, black_line_height=4, parity='even')
    # write_image(img, 'even.png', 'uint8')
    # img = create_interlace_mask_img(
    #     width=640, height=480, black_line_height=4, parity='odd')
    # write_image(img, 'odd.png', 'uint8')

    # color_mask_list = [
    #     [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    #     [1, 0, 1], [1, 1, 0], [0, 1, 1]]
    # black_line_height = 8
    # for color_mask in color_mask_list:
    #     create_interlaced_still(
    #         width=1920, height=1080, color_mask=color_mask,
    #         black_line_height=black_line_height)
    #     create_interlaced_still(
    #         width=320, height=240, color_mask=color_mask,
    #         black_line_height=1)

    # interlaced_still_for_debug_420(black_line_height=2)
    # create_dot_mesh_image(
    #     width=640, height=480, dot_size=[4, 2], st_offset=[2, 1])
    # offset_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # for offset in offset_list:
    #     img = tpg.create_dot_mesh_image(
    #         width=1920, height=1080, dot_size=[2, 2], st_offset=offset)
    #     img = img * np.array([1, 0, 1])
    #     fname = f"./img/dot_{offset[0]}-{offset[1]}.png"
    #     print(fname)
    #     write_image(img, fname, 'uint8')

    # offset_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # for offset in offset_list:
    #     analyze_ycbcr420_encode_each_offset(pix_fmt='yuv420p', offset=offset)
    #     analyze_ycbcr420_encode_each_offset(pix_fmt='yuv444p', offset=offset)

    # debug_x265()
    # debug_sample_interlaced_image()
    # comparison_420_444()
    debug_with_davinci_420()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
    # from colour import RGB_to_YCbCr, WEIGHTS_YCBCR
    # K_601 = WEIGHTS_YCBCR['ITU-R BT.601']
    # K_709 = WEIGHTS_YCBCR['ITU-R BT.709']
    # rgb = np.array([1, 0, 1])
    # # rgb = np.array([227, 9, 228])/255
    # ycbcr = RGB_to_YCbCr(
    #     rgb, K=K_709, in_int=False, out_int=True,
    #     in_legal=False, out_legal=True, out_bits=8)
    # print(ycbcr)
    # print(f"0x{ycbcr[0]:04X}, 0x{ycbcr[1]:04X}, 0x{ycbcr[2]:04X}")
