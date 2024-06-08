# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

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

    dst_fname = src_fname.replace(".jxr", ".exr")
    image = imread(src_fname)
    print(image.dtype)
    write_image(image=image, path=dst_fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func(src_fname="./Windows_HDR_Capture/DirectX/rgb_10bit.jxr")
    # main_func(src_fname="./Windows_HDR_Capture/DirectX/rgb_16bit.jxr")
    # main_func(src_fname="./Windows_HDR_Capture/gain_1.0/AW3225QF.jxr")
    # main_func(src_fname="./Windows_HDR_Capture/DirectX/Rec2100_10bit_to_scRGB.jxr")
    # main_func(src_fname="./Windows_HDR_Capture/DirectX/check_rec2020_to_rec709_matrix.jxr")
    # main_func(src_fname="./Windows_HDR_Capture/DirectX/FP16_Rec2020_to_Rec709.jxr")
    # main_func(src_fname="./Windows_HDR_Capture/DirectX/UINT10_Rec2020_to_Rec709.jxr")
    # main_func(src_fname="./Windows_HDR_Capture/DirectX/FP16_Rec2020_to_Rec709_half_mul_V8.jxr")
    main_func(src_fname="./Windows_HDR_Capture/DirectX/FP16_Rec2020_to_Rec709_V11.jxr")
    # main_func(src_fname="./Windows_HDR_Capture/DirectX/UINT10_Rec2020_to_Rec709_V2.jxr")
