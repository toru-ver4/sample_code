# -*- coding: utf-8 -*-
"""
debug
=====
"""

# import standard libraries
import os
from pathlib import Path
import subprocess

# import third-party libraries
import numpy as np
import cv2

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

STIL_DIR = "/work/overuse/2020/005_make_countdown_movie/ffmpeg_out/"
H264_STILS = [
    "420_h264_stil_0001.png",
    "422_h264_stil_0001.png",
    "444_h264_stil_0001.png",
    "422_ProRes_stil_0001.png",
    "444_ProRes_stil_0001.png"]

EXTRACT_ST_POS_H = 1592
EXTRACT_ST_POS_V = 244
EXTRACT_ED_POS_H = 1801
EXTRACT_ED_POS_V = 977


def check_dot_droped_pattern():
    """
    Encode and decode with some kind of chroma subsampoing.
    And check the result.
    """
    # encode
    subprocess.run(["./encode.sh"])

    # extract
    for still_base_name in H264_STILS:
        in_fname = os.path.join(STIL_DIR, still_base_name)
        in_fname_without_ext = Path(in_fname).with_suffix("")
        out_fname = f"{in_fname_without_ext}_trim.png"
        img = cv2.imread(in_fname)
        out_img = img[
            EXTRACT_ST_POS_V:EXTRACT_ED_POS_V,
            EXTRACT_ST_POS_H:EXTRACT_ED_POS_H]
        out_img = cv2.resize(
            out_img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(out_fname, out_img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    check_dot_droped_pattern()
