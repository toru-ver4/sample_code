# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import subprocess

# import third-party libraries
import numpy as np
from imagecodecs import JPEGXR, imread
from colour.io import write_image

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def avif_to_av1(src_fname="", dst_fname=""):
    cmd = "ffmpeg"
    ops = [
        "-framerate", "24",
        '-loop', "1",
        '-t', "0.5",
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-i', src_fname,
        '-c:v', 'libsvtav1',
        '-movflags', 'write_colr',
        '-pix_fmt', 'yuv444p10le',
        '-crf', '0',
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        dst_fname,
        '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    src_fname = "./debug/src_tp/HDR_TP_3840x960.png"
    dst_fname = "./debug/src_tp/HDR_TP_3840x960.mp4"
    avif_to_av1(src_fname=src_fname, dst_fname=dst_fname)
