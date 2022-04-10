# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from typing import List
from multiprocessing import Pool, cpu_count
from matplotlib.pyplot import text

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


class MovingSquareConcentricallyTP():
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.scale_coef = int(height / 1080 + 0.5)
        self.info_font_size = int(30 * self.scale_coef + 0.5)
        self.annotate_font_size = int(20 * self.scale_coef + 0.5)
        self.annotate_font_edge_size = int(self.annotate_font_size * 0.1 + 0.5)
        self.font_color = np.array([50, 50, 50]) / 100  # linear value
        self.annotate_font_color = np.array([50, 50, 50]) / 100  # linear value
        self.font = fc2.NOTO_SANS_MONO_BOLD
        self.base_period = 3
        self.num_of_period = 2
        self.square_color_list = np.array(
            [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8],
             [0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]])
        self.blink_fps_list = [0.5, 1, 2, 4, 6]
        self.fps = 60

        self.text_margin_rate = 0.2
        self.bg_color = np.array([0.03, 0.03, 0.03])
        self.info_text = " Thie is sample. Rev.01"
        self.create_bg_img()
        self.active_height = self.height - self.text_area_height
        self.square_size = int(self.active_height * 0.13)
        self.create_square_obj_list()

        # self.debug_print()

    def debug_print(self):
        print(f"self.active_height//2={self.active_height//2}")

    def draw_square_seq(self):
        num_of_frame = self.base_period * self.num_of_period * self.fps

        total_process_num = num_of_frame
        block_process_num = int(cpu_count() * 2)
        block_num = int(round(total_process_num / block_process_num + 0.5))
        for b_idx in range(block_num):
            args = []
            for p_idx in range(block_process_num):
                l_idx = b_idx * block_process_num + p_idx              # User
                print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
                if l_idx >= total_process_num:                         # User
                    break
                d = dict(frame_idx=l_idx)
                args.append(d)
                # self.draw_square_core(**d)
                # if l_idx > 3:
                #     break
                # break
            with Pool(block_process_num) as pool:
                pool.map(self.thread_wrapper_draw_square_core, args)
            # break

    def thread_wrapper_draw_square_core(self, args):
        self.draw_square_core(**args)

    def draw_tp(self, frame_idx, square_obj, img):
        square_obj.calc_pos(frame_idx)
        st_pos, ed_pos = square_obj.get_st_pos_ed_pos()
        modified_color = square_obj.get_modified_color()
        img[st_pos[1]:ed_pos[1], st_pos[0]:ed_pos[0]] = modified_color

    def draw_desc_text_with_param(self, img, square_obj, idx_v, idx_h):
        font_size = self.annotate_font_size
        font_color = self.annotate_font_color

        text = f"{self.blink_fps_list[idx_h]} fps"
        if idx_v == 0:
            text += " (gradually)"
        else:
            text += " (suddenly)"

        text_draw_ctrl = fc2.TextDrawControl(
            text=text, font_color=font_color, font_size=font_size,
            font_path=fc2.NOTO_SANS_CJKJP_MEDIUM,
            stroke_width=self.annotate_font_edge_size,
            stroke_fill=(0, 0, 0))
        _, text_height = text_draw_ctrl.get_text_width_height()
        lb_pos = square_obj.get_text_desc_lb_pos()
        pos = [
            lb_pos[0], lb_pos[1] - int(text_height * 1.2)]
        text_draw_ctrl.draw(img=img, pos=pos)

    def draw_square_core(self, frame_idx):
        img = self.img.copy()
        for idx_v in range(2):
            for idx_h in range(len(self.square_color_list)):
                square_obj = self.square_obj_list[idx_v][idx_h]
                self.draw_tp(
                    frame_idx=frame_idx, square_obj=square_obj, img=img)
                self.draw_desc_text_with_param(
                    img=img, square_obj=square_obj, idx_h=idx_h, idx_v=idx_v)

                # create file name
                size = square_obj.get_size()
                fname = self.create_file_name(
                    size=size, square_obj=square_obj,
                    width=self.width, height=self.height, frame_idx=frame_idx)
        print(fname)
        self.write_frame(fname=fname, img=img)

    def write_frame(self, fname, img):
        img_normalized = np.uint16(
            np.round(
                tf.oetf_from_luminance(
                    img*100, tf.ST2084) * FFMPEG_NORMALIZE_COEF))/0xFFFF
        # tpg.img_write(fname, img_int)
        write_image(image=img_normalized, path=fname, bit_depth='uint16')

    def create_file_name(
            self, square_obj, size=100,
            width=1920, height=1080, frame_idx=0):
        ext = ".png"
        base_dir = "/work/overuse/2022/flicker_square/"
        desc = "flicker"
        size_s = f"size-{size:04d}"
        fg_color = square_obj.get_color()
        bg_color = self.bg_color

        fg_c = f"fg-{fg_color[0]:.2f}-{fg_color[1]:.2f}-{fg_color[2]:.2f}"
        bg_c = f"bg-{bg_color[0]:.2f}-{bg_color[1]:.2f}-{bg_color[2]:.2f}"
        resolution = f"{width}x{height}"
        index = f"{frame_idx:04d}"
        base_name = "_".join([desc, size_s, fg_c, bg_c, resolution, index])
        fname = base_dir + base_name + ext

        return fname

    def create_square_obj_list(self):
        block_len_h = int(self.width / len(self.blink_fps_list) + 0.5)
        block_len_v = int(self.active_height / 2 + 0.5)

        self.square_obj_list = []
        for idx_v in range(2):
            st_pos_v = int(block_len_v * idx_v + (block_len_v//2) + 0.5)
            smooth = True if idx_v == 0 else False
            temp_list = []
            for idx_h in range(len(self.blink_fps_list)):
                st_pos_h = int(block_len_h * idx_h + (block_len_h//2) + 0.5)
                blink_fps = self.blink_fps_list[idx_h]
                square_obj = FlickerSquare(
                    size=self.square_size, color=self.square_color_list[idx_h],
                    blink_fps=blink_fps,
                    base_period=self.base_period,
                    num_of_period=self.num_of_period,
                    global_st_pos=[st_pos_h, st_pos_v],
                    fps=self.fps, smooth=smooth)
                temp_list.append(square_obj)
            self.square_obj_list.append(temp_list)

    def create_bg_img(self):
        self.img = np.ones((self.height, self.width, 3)) * self.bg_color
        text_area_width = self.width
        text_draw_ctrl = fc2.TextDrawControl(
            text=self.info_text, font_color=self.font_color,
            font_size=self.info_font_size, font_path=self.font,
            stroke_width=0, stroke_fill=None)
        _, text_height = text_draw_ctrl.get_text_width_height()
        self.text_area_height = int(
            text_height * (1 + self.text_margin_rate * 2) + 0.5)
        text_area_img = np.zeros((self.text_area_height, text_area_width, 3))
        text_pos = (
            0, (self.text_area_height // 2) - (text_height // 2))
        text_draw_ctrl.draw(img=text_area_img, pos=text_pos)
        tpg.merge(
            self.img, text_area_img, (0, self.height-self.text_area_height))


class FlickerSquare():
    def __init__(
            self, size=200, color=[0.1, 0.1, 0.0], blink_fps=1,
            base_period=2, num_of_period=4, global_st_pos=[0, 0],
            fps=60, smooth=False):
        self.size = size
        self.global_st_pos = global_st_pos
        self.base_color = np.array(color)
        self.blink_fps = blink_fps
        self.base_period = base_period
        self.num_of_period = num_of_period
        self.fps = fps
        self.num_of_frame = fps * self.base_period
        self.smooth = smooth

        # "lb" means left-bottom
        self.desc_text_lb_pos = [
            global_st_pos[0] - int(size / 2 + 0.5),
            global_st_pos[1] - int(size / 2 + 0.5)]

    def calc_pos(self, idx):
        y = np.sin(
            2*np.pi/(self.fps)*idx*self.blink_fps - np.pi/2) * 0.5 + 0.5
        if self.smooth is False:
            if y > 0.5:
                y = 1
            else:
                y = 0
        pos_h = -self.size // 2
        pos_v = -self.size // 2
        self.st_pos = [
            self.global_st_pos[0] + pos_h, self.global_st_pos[1] + pos_v]
        self.ed_pos = [
            self.st_pos[0] + self.size, self.st_pos[1] + self.size]
        color_st2084 = tf.oetf_from_luminance(self.base_color * 100, tf.ST2084)
        self.color = tf.eotf_to_luminance(color_st2084 * y, tf.ST2084) / 100

    def get_st_pos_ed_pos(self):
        # print(f"st, ed, co = {self.st_pos, self.ed_pos, self.color}")
        return self.st_pos, self.ed_pos

    def get_modified_color(self):
        return self.color

    def get_color(self):
        return self.base_color

    def get_size(self):
        return self.size

    def get_text_desc_lb_pos(self):
        return self.desc_text_lb_pos


def main_func():
    moving_square_tp = MovingSquareConcentricallyTP(width=1920, height=1080)
    moving_square_tp.draw_square_seq()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
