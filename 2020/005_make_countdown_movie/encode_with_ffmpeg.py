# -*- coding: utf-8 -*-
"""
音声ファイルを作る
===================
"""

# import standard libraries
import os
from itertools import product
import subprocess

# import third-party libraries

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

REVISION = 9
SRC_DIR = "D:/abuse/2020/005_make_countdown_movie/movie_seq/"
DST_DIR = "D:/abuse/2020/005_make_countdown_movie/ffmpeg_out/"


def create_input_file_name(width, height, fps, dr):
    fname = SRC_DIR + "movie_{:}_{:}x{:}_{:}fps_0000.png".format(
        dr, width, height, fps)
    fname = fname.replace("0000", r"%4d")

    return fname


def create_output_file_name(width, height, fps, dr, suffix='yuv444'):
    fname = DST_DIR + "countdown_{:}_{:}x{:}_{:}fps_Rev{:}_{:}.mov".format(
        dr, width, height, fps, REVISION, suffix)

    return fname


def encode_sdr_444(width, height, fps, dr):
    if dr == "HDR":
        return
    in_fname = create_input_file_name(width, height, fps, dr)
    in_fname_ffmpeg = in_fname.replace("0000", r"%4d")
    out_fname = create_output_file_name(
        width, height, fps, dr, suffix="hevc_yuv444p12le_qp-0")
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-r', str(fps), '-i', in_fname_ffmpeg,
        '-i', './wav/countdown.wav', '-c:a', 'aac',
        '-c:v', 'hevc',
        '-movflags', 'write_colr',
        '-pix_fmt', 'yuv444p12le', '-qp', '0',
        '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        str(out_fname), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def encode_sdr_420(width, height, fps, dr):
    if dr == "HDR":
        return
    in_fname = create_input_file_name(width, height, fps, dr)
    in_fname_ffmpeg = in_fname.replace("0000", r"%4d")
    out_fname = create_output_file_name(
        width, height, fps, dr, suffix="hevc_yuv420p10le_qp-0")
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-r', str(fps), '-i', in_fname_ffmpeg,
        '-i', './wav/countdown.wav', '-c:a', 'aac',
        '-c:v', 'hevc',
        '-movflags', 'write_colr',
        '-pix_fmt', 'yuv420p10le', '-qp', '0',
        '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        str(out_fname), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def encode_hdr_444(width, height, fps, dr):
    if dr == "SDR":
        return
    in_fname = create_input_file_name(width, height, fps, dr)
    in_fname_ffmpeg = in_fname.replace("0000", r"%4d")
    out_fname = create_output_file_name(
        width, height, fps, dr, suffix="hevc_yuv444p12le_qp-0")
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-r', str(fps), '-i', in_fname_ffmpeg,
        '-i', './wav/countdown.wav', '-c:a', 'aac',
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


def encode_hdr_420(width, height, fps, dr):
    if dr == "SDR":
        return
    in_fname = create_input_file_name(width, height, fps, dr)
    in_fname_ffmpeg = in_fname.replace("0000", r"%4d")
    out_fname = create_output_file_name(
        width, height, fps, dr, suffix="hevc_yuv420p10le_qp-0")
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-r', str(fps), '-i', in_fname_ffmpeg,
        '-i', './wav/countdown.wav', '-c:a', 'aac',
        '-c:v', 'hevc',
        '-movflags', 'write_colr',
        '-pix_fmt', 'yuv420p10le', '-qp', '0',
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        str(out_fname), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def encode_each_data():
    # resolution_list = [[1920, 1080], [2048, 1080], [4096, 2160], [3840, 2160]]
    # fps_list = [24, 30, 50, 60, 120]
    resolution_list = [[1920*4, 1080*4]]
    fps_list = [60]
    # dynamic_range_list = ['SDR', 'HDR']
    dynamic_range_list = ['SDR']
    for resolution, fps, dynamic_range in product(
            resolution_list, fps_list, dynamic_range_list):
        width = resolution[0]
        height = resolution[1]
        encode_sdr_444(
            width=width, height=height, fps=fps, dr=dynamic_range)
        # encode_hdr_444(
        #     width=width, height=height, fps=fps, dr=dynamic_range)
        encode_sdr_420(
            width=width, height=height, fps=fps, dr=dynamic_range)
        # encode_hdr_420(
        #     width=width, height=height, fps=fps, dr=dynamic_range)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    encode_each_data()
