# -*- coding: utf-8 -*-
"""
HDTVのカラーバーを作る
=====================

* とりあえず YPbPr で作る
* 最後に CbYCr or RGB に変換する感じでオナシャス！

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import YCbCr_to_RGB, YCBCR_WEIGHTS

# import my libraries
import cv2

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

# 座標系のパラメータ Pattern1 -- 3 の横方向
aa = 1920
bb = 1080
cc_total_num = 7
cc = int(3/4 * aa / cc_total_num)
dd = (aa - cc * cc_total_num) // 2 + 1

# 座標系のパラメータ Pattern4 の横方向
cc0b1 = int(3/2*cc) + 1
cc100w = 2 * cc + 1
cc0b2 = int(5/6*cc) + 1
ccpluge = int(1/3*cc)
cc0b3 = cc

# 座標系のパラメータ 縦方向
bb1 = int(7/12*bb)
bb2 = int(1/12*bb)
bb3 = int(1/12*bb)
bb4 = int(3/12*bb)

# 色のパラメータ
pt1_gray = np.array([414, 512, 512])
pt1_75w = np.array([721, 512, 512])
pt1_yy = np.array([674, 176, 543])
pt1_cc = np.array([581, 589, 176])
pt1_gg = np.array([534, 253, 207])
pt1_mm = np.array([251, 771, 817])
pt1_rr = np.array([204, 435, 848])
pt1_bb = np.array([111, 848, 481])

pt2_cc = np.array([754, 615, 64])
pt2_100w = np.array([940, 512, 512])
pt2_75w = np.array([721, 512, 512])
pt2_bb = np.array([127, 960, 471])

pt3_yy = np.array([877, 64, 553])
pt3_rr = np.array([250, 409, 960])
pt3_black = np.array([64, 512, 512])
pt3_white = np.array([940, 512, 512])

pt4_15g = np.array([195, 512, 512])
pt4_black = np.array([64, 512, 512])
pt4_100w = np.array([940, 512, 512])
pt4_n2b = np.array([46, 512, 512])
pt4_p2b = np.array([82, 512, 512])
pt4_p4b = np.array([99, 512, 512])


def make_pattern1():
    img_gray = np.ones((bb1, dd, 3)) * pt1_gray
    img_w75 = np.ones((bb1, cc, 3)) * pt1_75w
    img_yy = np.ones((bb1, cc, 3)) * pt1_yy
    img_cc = np.ones((bb1, cc, 3)) * pt1_cc
    img_gg = np.ones((bb1, cc, 3)) * pt1_gg
    img_mm = np.ones((bb1, cc, 3)) * pt1_mm
    img_rr = np.ones((bb1, cc, 3)) * pt1_rr
    img_bb = np.ones((bb1, cc, 3)) * pt1_bb

    return np.hstack([
        img_gray, img_w75, img_yy, img_cc, img_gg,
        img_mm, img_rr, img_bb, img_gray])


def make_pattern2():
    img_cc = np.ones((bb2, dd, 3)) * pt2_cc
    img_100w = np.ones((bb2, cc, 3)) * pt2_100w
    img_75w = np.ones((bb2, cc * 6, 3)) * pt2_75w
    img_bb = np.ones((bb2, dd, 3)) * pt2_bb

    return np.hstack([img_cc, img_100w, img_75w, img_bb])


def make_pattern3():
    ramp_num = 940 - 64 + 1
    img_yy = np.ones((bb3, dd, 3)) * pt3_yy
    img_rr = np.ones((bb3, dd, 3)) * pt3_rr
    cc_total_len = cc * cc_total_num
    cc_for_black = (cc_total_len - ramp_num) // 2
    cc_for_white = (cc_total_len - ramp_num) // 2

    # print(f"cc_total_len={}, sum={cc_for_black + cc_for_white}")
    img_ramp_black = np.ones((bb3, cc_for_black, 3)) * pt3_black
    img_ramp_white = np.ones((bb3, cc_for_white, 3)) * pt3_white
    img_ramp_y = np.arange(ramp_num) + 64
    img_ramp_cbcr = np.ones_like(img_ramp_y) * 512
    img_ramp = np.dstack((img_ramp_y, img_ramp_cbcr, img_ramp_cbcr))\
        * np.ones((bb3, ramp_num, 3))

    return np.hstack(
        [img_yy, img_ramp_black, img_ramp, img_ramp_white, img_rr])


def make_pattern4():
    img_15g = np.ones((bb4, dd, 3)) * pt4_15g
    img_0b1 = np.ones((bb4, cc0b1, 3)) * pt4_black
    img_100w = np.ones((bb4, cc100w, 3)) * pt4_100w
    img_0b2 = np.ones((bb4, cc0b2, 3)) * pt4_black
    img_n2b = np.ones((bb4, ccpluge, 3)) * pt4_n2b
    img_0bpluge = np.ones((bb4, ccpluge, 3)) * pt4_black
    img_p2b = np.ones((bb4, ccpluge, 3)) * pt4_p2b
    img_p4b = np.ones((bb4, ccpluge, 3)) * pt4_p4b
    img_0b3 = np.ones((bb4, cc0b3, 3)) * pt4_black

    return np.hstack([
        img_15g, img_0b1, img_100w, img_0b2, img_n2b, img_0bpluge, img_p2b,
        img_0bpluge, img_p4b, img_0b3, img_15g])


def save_images(img_ycbcr, img_rgb):
    """
    画像データを正規化後に16bit整数型として保存。
    正規化は 1023 が 1.0 となるようにマッピングしている。
    DaVinci Resolve の内部正規化ルールとマッチするように。
    """
    cv2.imwrite("HDTV_COLOR_BAR_YCbCr.tiff",
                np.uint16(img_ycbcr[..., ::-1] / 0x3FF * 0xFFFF))
    cv2.imwrite("HDTV_COLOR_BAR_RGB.tiff",
                np.uint16(img_rgb[..., ::-1] / 0x3FF * 0xFFFF))


def main_func():
    img_buf = []

    pattern1 = make_pattern1()
    img_buf.append(pattern1)

    pattern2 = make_pattern2()
    img_buf.append(pattern2)

    pattern3 = make_pattern3()
    img_buf.append(pattern3)

    pattern4 = make_pattern4()
    img_buf.append(pattern4)

    img_ycbcr = np.vstack(img_buf)

    # 横幅が1921px なので 一番右端を削る
    img_ycbcr = img_ycbcr[:, :-1, :]

    img_rgb = YCbCr_to_RGB(
        YCbCr=img_ycbcr, K=YCBCR_WEIGHTS['ITU-R BT.709'],
        in_bits=10, in_legal=True, in_int=True,
        out_bits=10, out_legal=True, out_int=True)

    save_images(img_ycbcr, img_rgb)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
