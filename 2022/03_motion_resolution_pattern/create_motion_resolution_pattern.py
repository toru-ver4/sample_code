# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
from multiprocessing.sharedctypes import Value
import os

# import third-party libraries
import numpy as np
from colour.io import write_image

# import my libraries
import font_control2 as fc2
import test_pattern_generator2 as tpg
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


FFMPEG_NORMALIZE_COEF = 65340
FFMPEG_NORMALIZE_COEF_INV = 65535/65340

IDMS_LINE_CV_LIST = [
    0, 128, 192, 0, 96, 144, 0, 64, 96]
IDMS_LINE_COLOR_LIST\
    = np.repeat(IDMS_LINE_CV_LIST, 3).reshape(len(IDMS_LINE_CV_LIST), 3)

IDMS_BG_CV_LIST = [255, 192, 128]
IDMS_BG_COLOR_LIST = np.repeat(IDMS_BG_CV_LIST, 3*3).reshape(-1, 3)


def conv_Nbit_to_linear(x, bit_depth=8, tf_str=tf.GAMMA24):
    """
    Examples
    --------
    >>> x = [255, 128, 64]
    >>> bit_depth = 8
    >>> tf_str = tf.ST2084
    >>> y = conv_8bit_to_linear(x, bit_depth, tf_str)
    >>> x2 = conv_linear_to_8bit(y, bit_depth, tf_str)
    >>> print(y)
    [  1.00000000e+02   9.40745992e-01   5.22570128e-02]
    >>> print(x2)
    [255 128  64]
    """
    max_value = (2 ** bit_depth) - 1
    y = tf.eotf_to_luminance(np.array(x)/max_value, tf_str)\
        / tf.REF_WHITE_LUMINANCE
    return y


def conv_linar_to_Nbit(x, bit_depth=8, tf_str=tf.GAMMA24):
    """
    Examples
    --------
    >>> x = [255, 128, 64]
    >>> bit_depth = 8
    >>> tf_str = tf.ST2084
    >>> y = conv_8bit_to_linear(x, bit_depth, tf_str)
    >>> x2 = conv_linear_to_8bit(y, bit_depth, tf_str)
    >>> print(y)
    [  1.00000000e+02   9.40745992e-01   5.22570128e-02]
    >>> print(x2)
    [255 128  64]
    """
    max_value = (2 ** bit_depth) - 1
    y = tf.oetf_from_luminance(x * tf.REF_WHITE_LUMINANCE, tf_str)
    y = np.round(y * max_value)

    if bit_depth <= 8:
        y_Nbit = np.uint8(y)
    elif bit_depth <= 16:
        y_Nbit = np.uint16(y)
    elif bit_depth <= 32:
        y_Nbit = np.uint32(y)
    else:
        y_Nbit = np.uint64(y)

    return y_Nbit


def conv_8bit_to_linear(x, tf_str=tf.GAMMA24):
    """
    Examples
    --------
    >>> x = [255, 128, 64]
    >>> tf_str = tf.ST2084
    >>> y = conv_8bit_to_linear(x, tf_str)
    >>> x2 = conv_linear_to_8bit(y, tf_str)
    >>> print(y)
    [  1.00000000e+02   9.40745992e-01   5.22570128e-02]
    >>> print(x2)
    [255 128  64]
    """
    y = conv_Nbit_to_linear(x, bit_depth=8, tf_str=tf_str)
    return y


def conv_linear_to_8bit(x, tf_str=tf.GAMMA24):
    """
    Examples
    --------
    >>> x = [255, 128, 64]
    >>> tf_str = tf.ST2084
    >>> y = conv_8bit_to_linear(x, tf_str)
    >>> x2 = conv_linear_to_8bit(y, tf_str)
    >>> print(y)
    [  1.00000000e+02   9.40745992e-01   5.22570128e-02]
    >>> print(x2)
    [255 128  64]
    """
    y = conv_linar_to_Nbit(x, bit_depth=8, tf_str=tf_str)

    return y


class LineUnit():
    """
    Methods
    -------
    * 
    """
    def __init__(
            self, width, height, line_width, num_of_lines,
            line_color1, line_color2, bg_color):
        """
        Parameters
        ----------
        width : int
            width
        height : int
            height
        line_width : int
            width of the line.
        num_of_lines : int
            number of the lines.
        line_color1 : ndarray
            line color value. It must be linear.
            (ex. np.array([0.5, 1.0, 0.9]))
        line_color2 : ndarray
            line color value. It must be linear.
            (ex. np.array([0.5, 1.0, 0.9]))
        bg_color : ndarray
            bg color value. It must be linear.
            (ex. np.array([0.5, 1.0, 0.9]))
        """
        self.width = width
        self.height = height
        self.line_width = line_width
        self.line_color1 = line_color1
        self.line_color2 = line_color2
        self.line_color_num = 2  # color1 and color2
        self.bg_color = bg_color

        self.create_bg_image()
        self.num_of_lines = num_of_lines
        self.calc_line_st_pos()
        self.draw()

    def calc_line_st_pos(self):
        line_area_width\
            = self.line_width * self.num_of_lines * self.line_color_num
        line_st_pos_h = self.width//2 - line_area_width//2
        line_st_pos_v = 0
        self.line_st_pos = [line_st_pos_h, line_st_pos_v]

    def create_bg_image(self):
        self.img = np.ones((self.height, self.width, 3)) * self.bg_color

    def draw(self):
        st_pos_h_base = self.line_st_pos[0]
        st_pos_v_base = self.line_st_pos[1]
        for idx in range(self.num_of_lines):
            st_pos_h\
                = st_pos_h_base + idx * self.line_width * self.line_color_num
            ed_pos_h = st_pos_h + self.line_width
            st_pos_v = st_pos_v_base
            ed_pos_v = st_pos_v_base + self.height
            self.img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = self.line_color1

            st_pos_h += self.line_width
            ed_pos_h += self.line_width
            self.img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = self.line_color2


class LineUnitBlock():
    def __init__(
            self, width, height, line_width, num_of_lines,
            line_color1_list, line_color2_list, bg_color_list):
        """
        Parameters
        ----------
        width : int
            width
        height : int
            height
        line_width : int
            width of the line.
        num_of_lines : int
            number of the lines.
        line_color1_list : ndarray
            line color value list. It must be linear.
            (ex. np.array([[0.5, 1.0, 0.9], [1.0, 0.8, 0.7], [1.0, 0.8, 0.7]]))
        line_color2_list : ndarray
            line color value list. It must be linear.
            (ex. np.array([[0.5, 1.0, 0.9], [1.0, 0.8, 0.7], [1.0, 0.8, 0.7]]))
        bg_color_list : ndarray
            bg color value list. It must be linear.
            (ex. np.array([[0.5, 1.0, 0.9], [1.0, 0.8, 0.7], [1.0, 0.8, 0.7]]))
        """
        self.width = width
        self.height = height
        self.line_width = line_width
        self.num_of_lines = num_of_lines
        self.line_color1_list = line_color1_list
        self.line_color2_list = line_color2_list
        self.bg_color_list = bg_color_list

        self.init_parameter_check()
        self.num_of_unit = len(line_color1_list)
        self.unit_width = self.width
        self.unit_height_list\
            = tpg.equal_devision(self.height, self.num_of_unit)
        self.draw()

    def init_parameter_check(self):
        if len(self.line_color1_list) != len(self.line_color2_list):
            msg = "line_color1 and line_color2 is not same lnegth."
            raise ValueError(msg)
        if len(self.line_color1_list) != len(self.bg_color_list):
            msg = "line_color1 and bg_color is not same lnegth."
            raise ValueError(msg)

    def draw(self):
        img_v_buf = []
        for idx in range(self.num_of_unit):
            lu = LineUnit(
                width=self.unit_width, height=self.unit_height_list[idx],
                line_width=self.line_width, num_of_lines=self.num_of_lines,
                line_color1=self.line_color1_list[idx],
                line_color2=self.line_color2_list[idx],
                bg_color=self.bg_color_list[idx])
            img_v_buf.append(lu.img)

        self.img = np.vstack(img_v_buf)


def debug_line_unit():
    width = 640
    height = 360
    line_width = 4
    num_of_lines = 5
    line_color1 = conv_8bit_to_linear([255, 128, 64])
    line_color2 = conv_8bit_to_linear([0, 0, 0])
    bg_color = conv_8bit_to_linear([64, 64, 64])
    lu = LineUnit(
        width=width, height=height, line_width=line_width,
        num_of_lines=num_of_lines,
        line_color1=line_color1, line_color2=line_color2, bg_color=bg_color)
    img_linear = lu.img

    out_img = conv_linear_to_8bit(img_linear)
    write_image(out_img/0xFF, "./hoge.png", 'uint8')


def debug_line_unit_block():
    width = 640
    height = 900
    line_width = 6
    num_of_lines = 8
    line_color1_list = conv_8bit_to_linear(IDMS_LINE_COLOR_LIST)
    bg_color_list = conv_8bit_to_linear(IDMS_BG_COLOR_LIST)
    line_color2_list = bg_color_list

    lub = LineUnitBlock(
            width=width, height=height, line_width=line_width,
            num_of_lines=num_of_lines,
            line_color1_list=line_color1_list,
            line_color2_list=line_color2_list,
            bg_color_list=bg_color_list)
    out_img = conv_linear_to_8bit(lub.img)
    write_image(out_img/0xFF, "./hoge.png", 'uint8')


def debug_func():
    # debug_line_unit()
    debug_line_unit_block()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
