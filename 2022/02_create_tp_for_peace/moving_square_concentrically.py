# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from typing import List
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


class MovingSquareConcentricallyTP():
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.scale_coef = int(height / 1080 + 0.5)
        self.info_font_size = int(30 * self.scale_coef + 0.5)
        self.annotate_font_size = int(20 * self.scale_coef + 0.5)
        self.font_color = np.array([50, 50, 50]) / 100  # linear value
        self.annotate_font_color = np.array([50, 50, 50]) / 100  # linear value
        self.font = fc2.NOTO_SANS_MONO_BOLD
        self.base_period = 3
        self.num_of_period = 2
        self.accel_rate_list = [2, 1.5, 1]
        self.square_color_list = np.array(
            [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]])
        self.square_pos_radius_rate_list = [0.3, 0.6, 0.9]
        self.fps = 60

        self.text_margin_rate = 0.2
        self.bg_color = np.array([0.0, 0.0, 0.0])
        self.square_num = len(self.square_color_list)
        self.info_text = " Thie is sample. Rev.01"
        self.create_bg_img()
        self.active_height = self.height - self.text_area_height
        self.square_size = int(self.active_height * 0.11)
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
                # break
            with Pool(block_process_num) as pool:
                pool.map(self.thread_wrapper_draw_square_core, args)
            # break

    def thread_wrapper_draw_square_core(self, args):
        self.draw_square_core(**args)

    def draw_square_core(self, frame_idx):
        img = self.img.copy()
        for obj_idx, square_obj in enumerate(self.square_obj_list):
            fg_color = self.square_color_list[obj_idx]
            bg_color = self.bg_color
            square_obj.calc_pos(frame_idx)
            st_pos, ed_pos = square_obj.get_st_pos_ed_pos()
            size = square_obj.get_size()
            img[st_pos[1]:ed_pos[1], st_pos[0]:ed_pos[0]] = fg_color
            fname = self.create_file_name(
                size=size, fg_color=fg_color, bg_color=bg_color,
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
            self, size=100, fg_color=[0.01, 0.01, 0.01],
            bg_color=0.1, width=1920, height=1080, frame_idx=0):
        ext = ".png"
        base_dir = "/work/overuse/2022/concentrically_square/"
        desc = "concentrically"
        size_s = f"size-{size:04d}"
        fg_c = f"fg-{fg_color[0]:.2f}-{fg_color[1]:.2f}-{fg_color[2]:.2f}"
        bg_c = f"bg-{bg_color[0]:.2f}-{bg_color[1]:.2f}-{bg_color[2]:.2f}"
        resolution = f"{width}x{height}"
        index = f"{frame_idx:04d}"
        base_name = "_".join([desc, size_s, fg_c, bg_c, resolution, index])
        fname = base_dir + base_name + ext

        return fname

    def create_square_obj_list(self):
        self.calc_square_margin_v()
        st_pos_h = self.width // 2
        st_pos_v = self.active_height // 2

        self.square_obj_list: List[SquareConcentrically] = []
        for idx in range(self.square_num):
            accel_rate = self.accel_rate_list[idx]
            radius_h =\
                self.width / 2 * self.square_pos_radius_rate_list[idx]\
                - (self.square_size // 2)
            radius_v =\
                self.active_height / 2 * self.square_pos_radius_rate_list[idx]\
                - (self.square_size // 2)
            square_obj = SquareConcentrically(
                size=self.square_size, color=self.square_color_list[idx],
                base_period=self.base_period, num_of_period=self.num_of_period,
                global_st_pos=[st_pos_h, st_pos_v],
                radius_h=radius_h, radius_v=radius_v,
                accel_rate=accel_rate, fps=self.fps)
            self.square_obj_list.append(square_obj)

    def calc_square_margin_v(self):
        square_margine = (
            self.active_height - self.square_size * self.square_num)\
            / (self.square_num + 1)
        self.square_margine = int(square_margine + 0.5)

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


class SquareConcentrically():
    def __init__(
            self, size=200, color=[0.1, 0.1, 0.0],
            base_period=2, num_of_period=4, global_st_pos=[0, 0],
            radius_h=200, radius_v=100,
            accel_rate=1.0, fps=60):
        self.size = size
        self.global_st_pos = global_st_pos
        self.color = np.array(color)
        self.base_period = base_period
        self.num_of_period = num_of_period
        self.accel_rate = accel_rate
        self.num_of_frame = fps * self.base_period
        self.radius_h = radius_h
        self.radius_v = radius_v

    def calc_pos(self, idx):
        y_h = np.cos(
            2*np.pi/(self.num_of_frame)*idx*self.accel_rate + np.pi)\
            * self.radius_h
        y_v = np.sin(
            2*np.pi/(self.num_of_frame)*idx*self.accel_rate + np.pi)\
            * self.radius_v
        pos_h = int(round(y_h)) - (self.size // 2)
        pos_v = int(round(y_v)) - (self.size // 2)
        self.st_pos = [
            self.global_st_pos[0] + pos_h,
            self.global_st_pos[1] + pos_v]
        self.ed_pos = [self.st_pos[0] + self.size, self.st_pos[1] + self.size]

    def get_st_pos_ed_pos(self):
        # print(f"st, ed = {self.st_pos, self.ed_pos}")
        return self.st_pos, self.ed_pos

    def get_size(self):
        return self.size


def main_func():
    moving_square_tp = MovingSquareConcentricallyTP()
    moving_square_tp.draw_square_seq()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
