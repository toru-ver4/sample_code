# -*- coding: utf-8 -*-
"""
fix alpha blending of font_control
===================================
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import font_control as fc


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def debug_alpha_blend(
        bg_color=[1.0, 0.0, 1.0], out_file_prefix="before_fix"):
    """
    Parameters
    ----------
    bg_level : float
        backgroud level. min=0.0(black), max=1.0(white)
    """
    width = 940
    height = 540
    bg_color = np.array(bg_color)
    fg_color = np.ones_like(bg_color) - bg_color
    img_gm24 = np.ones((height, width, 3)) * bg_color
    text_drawer = fc.TextDrawer(
        img_gm24, text="ABCDE, 1000cd/m2", pos=(50, 50),
        font_color=fg_color, font_size=50,
        bg_transfer_functions=tf.SRGB,
        fg_transfer_functions=tf.SRGB,
        font_path=fc.NOTO_SANS_MONO_BOLD)
    text_drawer.draw()

    bg_level_str = f"{bg_color[0]:.2f}-{bg_color[1]:.2f}-{bg_color[2]:.2f}"
    fname = f"./img/{out_file_prefix}_bg-{bg_level_str}_font_text.png"
    tpg.img_wirte_float_as_16bit_int(fname, img_gm24)
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_alpha_blend(bg_color=[0.0, 0.0, 0.0], out_file_prefix="before_fix")
    # debug_alpha_blend(bg_color=[1.0, 1.0, 1.0], out_file_prefix="before_fix")
    # debug_alpha_blend(bg_color=[1.0, 0.0, 1.0], out_file_prefix="before_fix")
    # debug_alpha_blend(bg_color=[0.0, 1.0, 0.0], out_file_prefix="before_fix")
    debug_alpha_blend(bg_color=[0.0, 0.0, 0.0], out_file_prefix="after_fix")
    debug_alpha_blend(bg_color=[1.0, 1.0, 1.0], out_file_prefix="after_fix")
    debug_alpha_blend(bg_color=[1.0, 0.0, 1.0], out_file_prefix="after_fix")
    debug_alpha_blend(bg_color=[0.0, 1.0, 0.0], out_file_prefix="after_fix")
