# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
from email.mime import base
import os
from pathlib import Path
from colour import read_image
from multiprocessing import Pool, cpu_count

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

REVISION = 1

IDMS_LINE_CV_LIST = [
    0, 128, 192, 0, 96, 144, 0, 64, 96]
IDMS_LINE_COLOR_LIST\
    = np.repeat(IDMS_LINE_CV_LIST, 3).reshape(len(IDMS_LINE_CV_LIST), 3)

IDMS_BG_CV_LIST = [255, 192, 128]
IDMS_BG_COLOR_LIST = np.repeat(IDMS_BG_CV_LIST, 3*3).reshape(-1, 3)

COLOR_MASK_LIST = [
    [1, 1, 1],
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 0, 1], [1, 1, 0], [0, 1, 1]
]


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


class UnitDescText():
    def __init__(
            self, width, height, line_width, font_color, bg_color, font_size):
        """
        Parameters
        ----------
        width : int
            width
        height : int
            height
        line_width int
            line width
        font_color : ndarray
            font color value. It must be linear.
            (ex. np.array([0.5, 1.0, 0.9]))
        bg_color : ndarray
            bg color value. It must be linear.
            (ex. np.array([0.5, 1.0, 0.9]))
        font_size : int
            font size
        """
        self.width = width
        self.height = height
        self.line_width = line_width
        self.font_color = font_color
        self.bg_color = bg_color
        self.font_size = font_size
        self.draw()

    def draw(self):
        text = f"Width: {self.line_width}px"
        img = np.ones((self.height, self.width, 3)) * self.bg_color
        text_draw_ctrl = fc2.TextDrawControl(
            text=text, font_color=self.font_color,
            font_size=self.font_size, font_path=fc2.NOTO_SANS_CJKJP_MEDIUM,
            stroke_width=0, stroke_fill=None)
        text_width, text_height = text_draw_ctrl.get_text_width_height()
        pos_h = (self.width // 2) - (text_width // 2)
        pos_v = (self.height // 2) - (text_height // 2)
        pos = (pos_h, pos_v)

        text_draw_ctrl.draw(img=img, pos=pos)
        self.img = img


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


def debug_unit_desc_text():
    font_color = conv_8bit_to_linear([128, 128, 128])
    bg_color = conv_8bit_to_linear([255, 255, 255])
    udt = UnitDescText(
        width=640, height=240, line_width=6,
        font_color=font_color, bg_color=bg_color, font_size=80)
    out_img = conv_linear_to_8bit(udt.img)
    write_image(out_img/0xFF, "./hoge_text.png", 'uint8')


def draw_text_info(
        img, width, height, text_area_height, font_size, text="hoge"):
    font_color = conv_8bit_to_linear([128, 128, 128])
    bg_color = conv_8bit_to_linear([0, 0, 0])
    text_img = np.ones((text_area_height, width, 3)) * bg_color
    text_draw_ctrl = fc2.TextDrawControl(
        text=text, font_color=font_color,
        font_size=font_size, font_path=fc2.NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)
    _, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = 0
    pos_v = (text_area_height // 2) - (text_height // 2)
    pos = (pos_h, pos_v)
    text_draw_ctrl.draw(img=text_img, pos=pos)

    # composite
    tpg.merge(img, text_img, pos=[0, height-text_area_height])


def create_vertical_block(
        width, height, line_width, num_of_lines,
        line_color1_list, line_color2_list, bg_color_list,
        unit_desc_height, font_size, font_color, font_bg_color):

    img_v_buf = []
    udt = UnitDescText(
        width=width, height=unit_desc_height, line_width=line_width,
        font_color=font_color, bg_color=font_bg_color, font_size=font_size)
    img_v_buf.append(udt.img)

    lub = LineUnitBlock(
            width=width, height=height, line_width=line_width,
            num_of_lines=num_of_lines,
            line_color1_list=line_color1_list,
            line_color2_list=line_color2_list,
            bg_color_list=bg_color_list)
    img_v_buf.append(lub.img)

    v_block_img = np.vstack(img_v_buf)

    return v_block_img


def create_file_name(width, height, color_mask, c_idx, revision):
    fname = f"./img/motion_resolution_pattern_{width}x{height}_"
    fname += f"c-{c_idx}-{color_mask[0]}-{color_mask[1]}-{color_mask[2]}_"
    fname += f"rev{revision}.png"

    return fname


def mask_color(src_color, mask):
    dst_color = src_color.copy()
    for idx in range(3):
        dst_color[..., idx] = src_color[..., idx] * mask[idx]

    return dst_color


def create_image(width=1920, height=1080):
    for color_mask_idx in range(len(COLOR_MASK_LIST)):
        create_image_each_color(
            width=width, height=height, color_mask_idx=color_mask_idx)


def create_image_each_color(
        width=1920, height=1080, color_mask_idx=0):
    color_mask = COLOR_MASK_LIST[color_mask_idx]
    size_ratio = height / 1080
    line_width_list = [1, 2, 3, 4, 6, 8, 10, 12, 14]
    num_of_line_width = len(line_width_list)
    num_of_lines = int(5 * size_ratio + 0.5)
    line_color1_list = conv_8bit_to_linear(IDMS_LINE_COLOR_LIST)
    line_color1_list = mask_color(line_color1_list, color_mask)
    bg_color_list = conv_8bit_to_linear(IDMS_BG_COLOR_LIST)
    bg_color_list = mask_color(bg_color_list, color_mask)
    line_color2_list = bg_color_list
    info_text_height_rate = 0.045
    unit_desc_text_height_rate = 0.10
    info_area_height = int(height * info_text_height_rate)
    unit_desc_height = int(height * unit_desc_text_height_rate)
    block_height = height - info_area_height - unit_desc_height
    block_width_list = tpg.equal_devision(width, num_of_line_width)
    font_color = conv_8bit_to_linear([16, 16, 16])
    font_bg_color = conv_8bit_to_linear([255, 255, 255])
    font_bg_color = mask_color(font_bg_color, color_mask)

    info_font_size = int(28 * size_ratio + 0.5)
    unit_desc_font_size = int(26 * size_ratio + 0.5)

    img = np.zeros((height, width, 3))
    info_text = f" Moving Picture Resolution Pattern, {width}x{height}, "
    info_text += f"Gamma 2.4, BT.709, D65, Rev.{REVISION}"
    draw_text_info(
        img=img, width=width, height=height, text_area_height=info_area_height,
        font_size=info_font_size, text=info_text)

    block_img_buf = []
    for idx in range(num_of_line_width):
        block_img = create_vertical_block(
            width=block_width_list[idx], height=block_height,
            line_width=line_width_list[idx], num_of_lines=num_of_lines,
            line_color1_list=line_color1_list,
            line_color2_list=line_color2_list,
            bg_color_list=bg_color_list,
            unit_desc_height=unit_desc_height,
            font_size=unit_desc_font_size,
            font_color=font_color, font_bg_color=font_bg_color)
        block_img_buf.append(block_img)
    block_img_all = np.hstack(block_img_buf)

    tpg.merge(img, block_img_all, [0, 0])

    img_8bit = conv_linear_to_8bit(img)
    fname = create_file_name(
        width=width, height=height,
        color_mask=color_mask, c_idx=color_mask_idx, revision=REVISION)
    print(fname)
    write_image(img_8bit/0xFF, fname, 'uint8')


def thread_wrapper_scroll_image_each_color_core(args):
    scroll_image_each_color_core(**args)


def scroll_image_each_color_core(
        width, src_fname, fname_base, scroll_px=4, f_idx=0):
    src_img = read_image(src_fname)
    temp_img = np.hstack([src_img, src_img, src_img])
    st_pos_h = scroll_px * f_idx
    ed_pos_h = st_pos_h + width
    crop_img = temp_img[:, st_pos_h:ed_pos_h]
    fname = fname_base.format(idx=f_idx)
    print(fname)
    write_image(crop_img, fname, 'uint8')


def scroll_image(width, height, scroll_px=4):
    for color_maxk_idx in range(len(COLOR_MASK_LIST)):
        scroll_image_each_color(
            width=width, height=height, color_mask_idx=color_maxk_idx,
            scroll_px=scroll_px)


def scroll_image_each_color(
        width, height, color_mask_idx, scroll_px=4):
    color_mask = COLOR_MASK_LIST[color_mask_idx]
    dst_dir = "/work/overuse/2022/03_motion_resolution_pattern/"
    src_fname = create_file_name(
        width=width, height=height,
        color_mask=color_mask, c_idx=color_mask_idx, revision=REVISION)
    pp = Path(src_fname)
    base_name = pp.stem
    ext = pp.suffix
    fname_base = f"{dst_dir}{base_name}_{scroll_px:02}px_{{idx:04}}{ext}"
    repeat_num = width * 2 // scroll_px

    total_process_num = repeat_num
    block_process_num = int(cpu_count() * 1.5)
    block_num = int(round(total_process_num / block_process_num + 0.5))
    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            d = dict(
                width=width, src_fname=src_fname,
                fname_base=fname_base, scroll_px=scroll_px, f_idx=l_idx)
            args.append(d)
            # scroll_image_each_color_core(**d)
            # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_scroll_image_each_color_core, args)
        # break


def main_func():
    # create_image(width=1920, height=1080)
    # create_image(width=3840, height=2160)
    # create_image(width=3840*2, height=2160*2)
    # scroll_px_list = [2, 4, 8]
    # scroll_px_list = [4]
    # scale_factor_list = [1, 2]
    # for scroll_px in scroll_px_list:
    #     for scale_factor in scale_factor_list:
    #         scroll_image(
    #             width=1920*scale_factor, height=1080*scale_factor,
    #             scroll_px=scroll_px)
    pass


def debug_func():
    # debug_line_unit()
    # debug_line_unit_block()
    debug_unit_desc_text()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_func()
    main_func()
