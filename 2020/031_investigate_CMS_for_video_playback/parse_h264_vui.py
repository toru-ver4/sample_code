# -*- coding: utf-8 -*-
"""
Parse VUI using h26x-extractor
==============================
MP4Box -raw 1 src_grad_tp_1920x1080_b-size_64_ffmpeg_HDR10.mp4 -out hoge.h264
"""

# import standard libraries
import os
import subprocess

# import third-party libraries

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def extract_h264_binary_stream(fname):
    cmd = "MP4Box"
    ops = ['-raw', '1', fname, '-out', fname + ".h264"]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def parse_NAL(fname):
    cmd = 'h26x-extractor'
    ops = ['-v', fname]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    extract_h264_binary_stream(
        fname="./video/src_grad_tp_1920x1080_b-size_64_ffmpeg_HDR10.mp4"
    )
    parse_NAL(
        fname="./video/src_grad_tp_1920x1080_b-size_64_ffmpeg_HDR10.mp4.h264")
