# -*- coding: utf-8 -*-
"""
System Gamma の確認パターンを作成する
====================================

方針
----
4枚あるいは16枚の画面をタイル状に並べる。
各タイルに1つの ColorChecker およびテキストでInfo を載せる。



"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.models import BT2020_COLOURSPACE
import cv2

# import my libraries
import test_pattern_generator2 as tpg
import font_control as fcl
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


INFO_TEXT_LUMINANCE = 18
OUTLINE_COLOR_LUMINANCE = 18
INFO_TEXT_COLOR = [
    tf.oetf_from_luminance(INFO_TEXT_LUMINANCE, tf.ST2084) for x in range(3)]
OUTLINE_COLOR = [tf.oetf_from_luminance(OUTLINE_COLOR_LUMINANCE, tf.ST2084)
                 for x in range(3)]


def make_color_checker_image_with_system_gamma(
        width=1920, height=1080, luminance=100, system_gamma=1.0):
    """
    Make a ColorChecker image with system-gamma applied.
    """
    color_checker_width = int(width * 0.4)
    font_st_pos = np.uint16(np.array([width, width]) * 0.03)
    font_size = int(width * 0.021)

    # color checker の RGB(Linear) 算出
    img = np.zeros((height, width, 3))
    linear_rgb = tpg.generate_color_checker_rgb_value(
        color_space=BT2020_COLOURSPACE, target_white=tpg.D65_WHITE)

    # System Gamma 摘要 ＆ Luminance 変更して OETF 摘要
    system_gamma_rgb = linear_rgb ** system_gamma
    lumiannce_rgb = system_gamma_rgb * luminance
    rgb_oetf = tf.oetf_from_luminance(
        np.clip(lumiannce_rgb, 0.0, luminance), tf.ST2084)
    color_checker_image = tpg.make_color_checker_image(
        rgb_oetf, width=color_checker_width, padding_rate=0.01)

    # Info Text 付与
    text = f"Peak Luminance = {luminance}cd/m2, " \
        + f"System Gamma = {system_gamma:.1f}"
    text_drawer = fcl.TextDrawer(
        img, text=text, pos=font_st_pos,
        font_color=INFO_TEXT_COLOR, font_size=font_size,
        bg_transfer_functions=tf.ST2084,
        fg_transfer_functions=tf.ST2084)
    text_drawer.draw()

    # 合成
    merge_pos = tpg.calc_st_pos_for_centering(
        bg_size=tpg.get_size_from_image(img),
        fg_size=tpg.get_size_from_image(color_checker_image))
    tpg.merge(img, color_checker_image, merge_pos)
    tpg.draw_outline(img, fg_color=OUTLINE_COLOR, outline_width=1)

    # cv2.imwrite("./test2.tiff", np.uint16(np.round(img[..., ::-1] * 0xFFFF)))

    return img


def main_func():
    """
    全体統括
    """
    img1 = make_color_checker_image_with_system_gamma(960, 540, 100, 1.0)
    img2 = make_color_checker_image_with_system_gamma(960, 540, 1000, 1.0)
    img3 = make_color_checker_image_with_system_gamma(960, 540, 100, 1.0)
    img4 = make_color_checker_image_with_system_gamma(960, 540, 1000, 1.2)

    img = np.vstack([np.hstack([img1, img2]), np.hstack([img3, img4])])

    cv2.imwrite("./100vs1000_3.tiff",
                np.uint16(np.round(img[..., ::-1] * 0xFFFF)))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
