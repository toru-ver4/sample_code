# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path

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


def main_func(src_fname="./Windows_HDR_Capture/600.jxr"):
    if not JPEGXR.available:
        print("JPEG XR is not supported")
        return

    base_name = Path(src_fname).stem
    dst_fname = f"./debug/jpeg_xr_to_exr/{base_name}.exr"
    image = imread(src_fname)
    print(f"{src_fname} --> {dst_fname}")
    write_image(image=image, path=dst_fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func(src_fname="./debug/jpeg_xr/Rec2100_PQ_TP.jxr")
    # main_func(src_fname="./debug/jpeg_xr/Rec2100_PQ_TP_with_BG.jxr")
    # main_func(src_fname="./debug/jpeg_xr/check_hdr_ref_white_80nits.jxr")
    # main_func(src_fname="./debug/jpeg_xr/check_hdr_ref_white_200nits.jxr")
    # main_func(src_fname="./debug/jpeg_xr/check_hdr_ref_white_204nits.jxr")
    # main_func(src_fname="./debug/jpeg_xr/check_hdr_ref_white_sdr80nits.jxr")
    # main_func(src_fname="./debug/jpeg_xr/check_hdr_ref_white_sdr100nits.jxr")
    main_func(src_fname="./debug/jpeg_xr/Lightroom_204nits.jxr")

    # fname_list = [
    #     "./debug/jpeg_xr/TP_080_nits.jxr",
    #     "./debug/jpeg_xr/TP_100_nits.jxr",
    #     "./debug/jpeg_xr/TP_140_nits.jxr",
    #     "./debug/jpeg_xr/TP_200_nits.jxr",
    #     "./debug/jpeg_xr/TP_204_nits.jxr",
    #     "./debug/jpeg_xr/TP_280_nits.jxr",
    #     "./debug/jpeg_xr/TP_480_nits.jxr",
    # ]
    # for fname in fname_list:
    #     main_func(src_fname=fname)
