# -*- coding: utf-8 -*-
"""
3DLUTを適用してブログ用の画像を作る
==================================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import read_LUT, write_image, read_image
import cv2

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def apply_hdr10_to_turbo_3dlut(src_img_name, dst_img_name, lut_3d_name):
    """
    HDR10の静止画に3DLUTを適用して Turbo の輝度マップを作る。
    """
    hdr_img = read_image(src_img_name)
    lut3d = read_LUT(lut_3d_name)
    luminance_map_img = lut3d.apply(hdr_img)
    write_image(luminance_map_img, dst_img_name, bit_depth='uint16')


def combine_image(src_list, dst_img_name):
    print(src_list)
    img_list = [cv2.imread(fname, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                for fname in src_list]
    img_list = [cv2.resize(img, dsize=(img.shape[1]//2, img.shape[0]//2))
                for img in img_list]
    img_v0 = np.hstack((img_list[0], img_list[1]))
    img_v1 = np.hstack((img_list[2], img_list[3]))
    img_out = np.vstack((img_v0, img_v1))

    cv2.imwrite(dst_img_name, img_out)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # apply_hdr10_to_turbo_3dlut(
    #     src_img_name="./figure/step_ramp.tiff",
    #     dst_img_name="./figure/3dlut_sample_turbo.png",
    #     lut_3d_name="./3dlut/PQ_BT2020_to_Turbo_sRGB.cube")
    # src_list = [
    #     "./img/umi_boost.tif",
    #     "./figure/umi_boost_before.png",
    #     "./figure/LuminanceMap_for_ST2084_BT2020_D65_MapRange_100-1000nits_65x65x65_umi_boost.png",
    #     "./figure/CodeValueMap_for_ST2084_MapRange_100-1000nits_65x65x65_umi_boost.png",
    # ]
    # combine_image(src_list, dst_img_name="./figure/combine_umi_boost.png")

    src_list = [
        "./img/src_riku.tif",
        "./figure/src_riku_before.png",
        "./figure/LuminanceMap_for_ST2084_BT2020_D65_MapRange_100-1000nits_65x65x65_src_riku.png",
        "./figure/CodeValueMap_for_ST2084_MapRange_100-1000nits_65x65x65_src_riku.png",
    ]
    combine_image(src_list, dst_img_name="./figure/combine_src_riku.png")

    src_list = [
        "./img/step_ramp.tiff",
        "./figure/step_ramp_before.png",
        "./figure/LuminanceMap_for_ST2084_BT2020_D65_MapRange_100-1000nits_65x65x65_step_ramp.png",
        "./figure/CodeValueMap_for_ST2084_MapRange_100-1000nits_65x65x65_step_ramp.png",
    ]
    combine_image(src_list, dst_img_name="./figure/combine_step_ramp.png")
