# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np
from colour.io import write_image, read_image
from sympy import arg

# import my libraries
import font_control2 as fc2
import test_pattern_generator2 as tpg
import transfer_functions as tf
import cv2

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


def main_func():
    pass


def calc_1dim_sine_curve(x, num_of_frame_one_cycle, accel_rate=1):
    """
    Calculate a one-dimensional sine curve.

    Parameters
    ----------
    x : int or ndarray (int)
        A frame index
    num_of_frame_one_cycle : int
        Number of frames in one cycle.
    accel_rate : float
        A accel rate.
        For example, accel_rate = 2 means 2x-speed.

    Examples
    --------
    >>> num_of_frame_one_cycle = 8
    >>> x = np.arange(num_of_frame_one_cycle * 2)
    >>> y = calc_1dim_sine_curve(
    ...     x=x, num_of_frame_one_cycle=num_of_frame_one_cycle, accel_rate=1)
    >>> for idx, value in enumerate(y):
    ...     print(idx, value)
    0 0.0
    1 0.146446609407
    2 0.5
    3 0.853553390593
    4 1.0
    5 0.853553390593
    6 0.5
    7 0.146446609407
    8 0.0
    9 0.146446609407
    10 0.5
    11 0.853553390593
    12 1.0
    13 0.853553390593
    14 0.5
    15 0.146446609407
    """
    y = np.sin(
        2*np.pi/(num_of_frame_one_cycle)*x*accel_rate - np.pi/2)*0.5 + 0.5
    return y


class SquareCoordinateMovingHorisontally():
    def __init__(
            self, size=200, global_st_pos=[0, 0], locus_len=512,
            base_period=2, num_of_period=4,
            base_fps=120, render_fps=60, accel_rate=1.0):
        """
        Parameters
        ----------
        size : int
            square size. unit is px.
        global_st_pos : int
            The start coordinate of the square.
            Note: this value is **center** of the square.
        locus_len : int
            length of the locus
        base_period : float
            one period. unit is second.
        num_of_period : int
            number of the repeat.
            ex: if base_period is 2 and num_of_period is 3,
                then total period is 2 x 3 = 6 [s]
        base_fps : int
            frame rate of the moving picture
        render_fps : int
            actual frame rate of the moving square.

        Note
        ----
        self.ul_st_pos means "upper left" pos
        """
        self.size = size
        self.global_st_pos = global_st_pos
        self.base_period = base_period
        self.num_of_period = num_of_period
        self.base_fps = base_fps
        self.render_fps = render_fps
        self.accel_rate = accel_rate
        self.num_of_frame_one_cycle = int(base_fps * self.base_period)
        self.locus_len = locus_len

    def calc_index_mask(self):
        """
        Calculate the index mask using base_fps and render_fps.
        """
        rate = self.base_fps // self.render_fps

        if rate == 1:
            mask = 0xFFFF
        elif rate == 2:
            mask = 0xFFFE
        elif rate == 4:
            mask = 0xFFFC
        elif rate == 8:
            mask = 0xFFF8
        else:
            mask = 0xFFFF

        return mask

    def calc_pos_h(self, idx):
        """
        Calculate the coordinate of the square.
        `self.ul_st_pos` means "upper left" position.
        `self.rb_ed_pos` means "upper left" position.
        """
        index_mask = self.calc_index_mask()
        masked_idx = idx & index_mask
        y = calc_1dim_sine_curve(
            x=masked_idx, num_of_frame_one_cycle=self.num_of_frame_one_cycle,
            accel_rate=self.accel_rate)
        self.pos_h = int(y * self.locus_len + 0.5)
        self.ul_st_pos = [
            self.global_st_pos[0] + self.pos_h - (self.size//2),
            self.global_st_pos[1] - (self.size//2)]
        self.rb_ed_pos = [
            self.ul_st_pos[0] + self.size, self.ul_st_pos[1] + self.size]

    def get_st_pos_ed_pos(self):
        return self.ul_st_pos, self.rb_ed_pos

    def get_size(self):
        return self.size


class MovingSquareImageMultiFrameRate():
    def __init__(
            self, square_img, square_name_prefix,
            width, height, bg_color,
            locus_len_rate=0.9, base_pediod=2, num_of_period=2,
            base_fps=120, render_fps=60, accel_rate=1,
            font_size=40, font_color=[0.3, 0.3, 0.3]):
        """
        Parameters
        ----------
        square_img : ndarray
            square image.
        square_name_prefix : str
            square name prefix for dst file name.
        width : int
            width of the entire image
        height : int
            height of the entire image
        bg_color : ndarray
            bg color. it must be linear.
            ex. np.array([1.0, 0.8, 0.1])
        locus_len_rate : float
            a parameter for calculate the locus_len
        base_period : float
            one period. unit is second.
        num_of_period : int
            number of the repeat.
            ex: if base_period is 2 and num_of_period is 3,
                then total period is 2 x 3 = 6 [s]
        base_fps : int
            frame rate of the moving picture
        render_fps : int
            actual frame rate of the moving square.
        font_size : int
            font size of the fps-info.
        font_color : list
            font color. It must be linear.
        """
        self.square_img = square_img
        self.square_size = square_img.shape[0]
        self.square_name_prefix = square_name_prefix
        self.base_fps = base_fps
        self.render_fps = render_fps
        self.total_frame = base_pediod * num_of_period * base_fps
        self.img_base = np.ones((height, width, 3)) * bg_color
        self.draw_fps_info(
            img=self.img_base, font_size=font_size, font_color=font_color)
        locus_len = int((width - self.square_size) * locus_len_rate)
        g_st_pos = [(width - locus_len)//2, height//2]
        self.square_pos_obj = SquareCoordinateMovingHorisontally(
            size=self.square_size, global_st_pos=g_st_pos,
            locus_len=locus_len, base_period=base_pediod,
            num_of_period=num_of_period, base_fps=base_fps,
            render_fps=render_fps, accel_rate=accel_rate)

    def draw_fps_info(self, img, font_size, font_color):
        text = f"  {self.render_fps}fps"
        text_draw_ctrl = fc2.TextDrawControl(
            text=text, font_color=font_color,
            font_size=font_size, font_path=fc2.NOTO_SANS_CJKJP_BLACK,
            stroke_width=int(font_size*0.2), stroke_fill=(0, 0, 0))
        _, text_height = text_draw_ctrl.get_text_width_height()

        text_draw_ctrl.draw(img=img, pos=[0, int(text_height/4)])

    def draw_sequence_core(self, idx):
        img = self.draw_each_frame(idx=idx)
        self.save_frame(idx=idx, img=img)

    def thread_wrapper_draw_sequence_core(self, args):
        self.draw_sequence_core(**args)

    def draw_sequence(self):
        """

        """
        # for idx in range(self.total_frame):
        total_process_num = self.total_frame
        block_process_num = int(cpu_count() * 0.8)
        block_num = int(round(total_process_num / block_process_num + 0.5))
        for b_idx in range(block_num):
            args = []
            for p_idx in range(block_process_num):
                l_idx = b_idx * block_process_num + p_idx              # User
                print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
                if l_idx >= total_process_num:                         # User
                    break
                d = dict(idx=l_idx)
                args.append(d)
                # self.draw_sequence_core(**d)
            with Pool(block_process_num) as pool:
                pool.map(self.thread_wrapper_draw_sequence_core, args)

    def draw_each_frame(self, idx):
        img = self.img_base.copy()
        self.square_pos_obj.calc_pos_h(idx=idx)
        st_pos, ed_pos = self.square_pos_obj.get_st_pos_ed_pos()
        img[st_pos[1]:ed_pos[1], st_pos[0]:ed_pos[0]] = self.square_img

        return img

    def save_frame(self, idx, img):
        dst_dir = "/work/overuse/2022/04_120Hz_tp/"
        fname = f"{dst_dir}{self.square_name_prefix}_"
        fname += f"{self.render_fps}-on-{self.base_fps}fps_{idx:04d}.png"
        print(fname)
        write_image(img, fname)


def debug_dot_pattern(nn=4, mag_rate=8):
    fg_color = np.array([1, 1, 1])
    bg_color = np.array([0, 0, 0])
    fname = f"./img/coplex_dot.png"
    img = tpg.complex_dot_pattern2(
        nn=nn, fg_color=fg_color, bg_color=bg_color, mag_rate=mag_rate)
    write_image(img, fname)


def debug_line_cross_pattern():
    fg_color = np.array([0, 0, 0])
    bg_color = np.array([1, 1, 1])
    img = tpg.line_cross_pattern(
        nn=3, num_of_min_line=1, fg_color=fg_color, bg_color=bg_color,
        mag_rate=32)

    write_image(img, "./img/line_cross.png")


def debug_multi_border_pattern():
    fg_color = np.array([1, 1, 1])
    bg_color = np.array([0.1, 0.1, 0.1])
    # length = 12
    # thickness = 2
    # img = np.zeros((length, length, 3))
    # tpg.draw_border_line(img, (0, 0), length, thickness, fg_color)
    # write_image(img, "./img/border_test.png", 'uint8')
    # ll = tpg.calc_l_for_block(
    #     block_idx=2, num_of_block=3, num_of_line=3)
    # print(ll)
    img = tpg.create_multi_border_tp(
        num_of_line=1, num_of_block=5,
        fg_color=fg_color, bg_color=bg_color, mag_rate=3)
    write_image(img, "./img/multi_border_tp_test.png", 'uint16')


def debug_MovingSquareImageMultiFrameRate():
    square_img = read_image("./img/multi_border_tp_test.png")
    square_name_prefix = 'multi_border_tp'
    width = 1920
    height = 1080 // 2
    bg_color = np.array([0.1, 0.1, 0.1])
    locus_len_rate = 0.9
    base_pediod = 2
    num_of_period = 2
    base_fps = 60
    render_fps1 = 30
    render_fps2 = 60
    fps_info_font_size = 40
    fps_info_color = np.array([0.3, 0.3, 0.3])
    accel_rate = 1
    common_params = dict(
        square_img=square_img, square_name_prefix=square_name_prefix,
        width=width, height=height, bg_color=bg_color,
        locus_len_rate=locus_len_rate, base_pediod=base_pediod,
        num_of_period=num_of_period, base_fps=base_fps, accel_rate=accel_rate,
        font_size=fps_info_font_size, font_color=fps_info_color)
    msimfr1 = MovingSquareImageMultiFrameRate(
        render_fps=render_fps1, **common_params)
    msimfr2 = MovingSquareImageMultiFrameRate(
        render_fps=render_fps2, **common_params)
    img1 = msimfr1.draw_each_frame(idx=1)
    img2 = msimfr2.draw_each_frame(idx=1)
    img = np.vstack([img1, img2])

    write_image(img, "./img/concat.png")


def create_and_composite_info_text_image(
        img, width, height, text, font_size, bg_color, fg_color):
    """
    Parameters
    ----------
    width : int
        width of the text area image
    height : int
        height of the text area image
    text : str
        text (written at bottom)
    font_size : int
        font size
    bg_color : ndarray
        bg color. It must be linear.
    fg_color : ndarray
        fg color. It must be linear.
    moving_square_height_list : int
        height list of moving squares
    fps_list : int
        fps list.
    """
    # draw info text
    text_img = np.ones((height, width, 3)) * bg_color
    text_draw_ctrl = fc2.TextDrawControl(
        text=text, font_color=fg_color,
        font_size=font_size, font_path=fc2.NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)

    _, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = 0
    pos_v = (height // 2) - (text_height // 2)
    pos = (pos_h, pos_v)

    text_draw_ctrl.draw(img=text_img, pos=pos)
    tpg.merge(img, text_img, pos=[0, img.shape[0] - height])


def thread_wrapper_render_and_save(args):
    render_and_save(**args)


def render_and_save(
        idx, img, msimfr1, msimfr2,
        name_prefix, base_fps, render_fps1, render_fps2):
    img1 = msimfr1.draw_each_frame(idx=idx)
    img2 = msimfr2.draw_each_frame(idx=idx)
    square_img = np.vstack([img1, img2])
    tpg.merge(img, square_img, (0, 0))
    dst_dir = "/work/overuse/2022/04_120Hz_tp/2nd_sample/"
    fname = f"{dst_dir}{name_prefix}_"
    fname += f"{render_fps1}P-{render_fps2}P-on-{base_fps}fps_{idx:04d}.png"
    print(fname)
    write_image(img, fname)


def create_moving_square_final_image(
        square_fname="./img/multi_border_tp_test.png",
        name_prefix="multi_border",
        base_fps=120, render_fps1=60, render_fps2=120):
    width = 1920
    height = 1080
    # fg_color = np.array([1, 1, 1])
    text_info_fg_color = np.array([0.3, 0.3, 0.3])
    text_info_bg_color = np.array([0, 0, 0])
    bg_color = np.array([0.05, 0.05, 0.05])
    info_text_rate = 0.05
    font_size = 28

    # pattern parameters
    text = f"  Comparison of {render_fps1}P and {render_fps2}P"
    square_img = read_image(square_fname)
    locus_len_rate = 0.9
    square_name_prefix = 'multi_border_tp'

    # time parameters
    base_pediod = 2
    num_of_period = 1
    accel_rate = 1
    total_frame = base_fps * base_pediod * num_of_period

    info_text_height = int(height * info_text_rate)
    active_height = height - info_text_height
    height_list = tpg.equal_devision(active_height, 2)
    img = np.ones((height, width, 3)) * bg_color
    create_and_composite_info_text_image(
        img=img, width=width, height=info_text_height,
        text=text, font_size=font_size,
        bg_color=text_info_bg_color, fg_color=text_info_fg_color)

    common_params = dict(
        square_img=square_img, square_name_prefix=square_name_prefix,
        width=width, bg_color=bg_color,
        locus_len_rate=locus_len_rate, base_pediod=base_pediod,
        num_of_period=num_of_period, base_fps=base_fps, accel_rate=accel_rate)
    msimfr1 = MovingSquareImageMultiFrameRate(
        render_fps=render_fps1, height=height_list[0], **common_params)
    msimfr2 = MovingSquareImageMultiFrameRate(
        render_fps=render_fps2, height=height_list[1], **common_params)

    total_process_num = total_frame
    block_process_num = int(cpu_count() * 0.8)
    block_num = int(round(total_process_num / block_process_num + 0.5))
    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            d = dict(
                idx=l_idx, img=img, msimfr1=msimfr1, msimfr2=msimfr2,
                name_prefix=name_prefix, base_fps=base_fps,
                render_fps1=render_fps1, render_fps2=render_fps2)
            args.append(d)
        #     render_and_save(**d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_render_and_save, args)


def debug_func():
    # debug_dot_pattern(nn=5, mag_rate=8)
    # debug_line_cross_pattern()
    # debug_multi_border_pattern()
    # debug_MovingSquareImageMultiFrameRate()
    square_fname = "./img/coplex_dot.png"
    name_prefix = "complex_dot"
    create_moving_square_final_image(
        square_fname=square_fname, name_prefix=name_prefix,
        base_fps=120, render_fps1=60, render_fps2=120)
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
    # main_func()
    # num_of_frame_one_cycle = 8
    # x = np.arange(num_of_frame_one_cycle * 2)
    # y = calc_1dim_sine_curve(
    #     x=x, num_of_frame_one_cycle=num_of_frame_one_cycle, accel_rate=1)
    # for idx, value in enumerate(y):
    #     print(idx, value)
