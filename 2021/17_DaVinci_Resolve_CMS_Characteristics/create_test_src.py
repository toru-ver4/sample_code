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

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


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


def create_src_exr_img_seq():
    fps = 24
    sec = 1
    width = 1920
    height = 1080
    min_exposure = EXR_MIN_EXPOSURE
    max_exposure = EXR_MAX_EXPOSURE
    x = np.linspace(0, 1, width)
    line = tpg.shaper_func_log2_to_linear(
        x, min_exposure=min_exposure, max_exposure=max_exposure)
    img = tpg.h_mono_line_to_img(line, height)

    frame = fps * sec

    for idx in range(frame):
        fname = get_media_src_fname_exr(idx=idx)
        print(fname)
        write_image(
            image=img, path=fname, bit_depth='float32', method="OpenImageIO")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_src_png_img_seq()
    create_src_exr_img_seq()
