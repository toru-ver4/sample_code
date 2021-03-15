# -*- coding: utf-8 -*-
"""
fix alpha blending of font_control
===================================
"""

# import standard libraries
import os
from pathlib import Path
from types import FunctionType

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg
from test_pattern_coordinate import GridCoordinate, ImgWithTextCoordinate
import font_control as fc

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


IMG_SEQ_DIR = Path("/work/overuse/2021/02_8bt_10bit/img_seq")
REVISION = 1


def create_and_save_8bit_10bit_patch(
        width=512, height=1024, total_step=20, direction='h', level='middle'):
    img_out_8bit, img_out_10bit = tpg.create_8bit_10bit_id_patch(
        width=width, height=height, total_step=total_step,
        direction=direction, level=level)

    save_8bit_10bit_patch(
        img_8bit=img_out_8bit, img_10bit=img_out_10bit,
        width=width, height=height, total_step=total_step,
        direction=direction, level=level)


def save_8bit_10bit_patch(
        img_8bit, img_10bit,
        width=512, height=1024, total_step=20, direction='h', level='middle'):

    name_8bit\
        = f"./img/8bit_grad_{width}x{height}_dir_{direction}_{level}.png"
    name_10bit\
        = f"./img/10bit_grad_{width}x{height}_dir_{direction}_{level}.png"
    tpg.img_wirte_float_as_16bit_int(name_8bit, img_8bit)
    tpg.img_wirte_float_as_16bit_int(name_10bit, img_10bit)


def create_slide_seq(
        width=512, height=512, total_step=20, level='low'):
    img_out_8bit, img_out_10bit = tpg.create_8bit_10bit_id_patch(
        width=width, height=height, total_step=total_step,
        direction='h', level=level)


def id_patch_generator_class_test(
        width=512, height=512, total_step=30, level='low', step=2):
    generator = tpg.IdPatch8bit10bitGenerator(
        width=width, height=height, total_step=total_step, level=level,
        slide_step=step)
    frame_num = 180
    fname_8bit_base = "img_8bit_{width}x{height}_{step}step_{div}div_"
    fname_8bit_base += "{level}_{idx:04d}.png"
    fname_8bit_base = str(IMG_SEQ_DIR / fname_8bit_base)
    fname_10bit_base = "img_10bit_{width}x{height}_{step}step_{div}div_"
    fname_10bit_base += "{level}_{idx:04d}.png"
    fname_10bit_base = str(IMG_SEQ_DIR / fname_10bit_base)

    for idx in range(frame_num):
        img_8bit, img_10bit = generator.extract_8bit_10bit_img()
        fname_8bit = fname_8bit_base.format(
            width=width, height=height, step=step, div=total_step,
            level=level, idx=idx)
        fname_10bit = fname_10bit_base.format(
            width=width, height=height, step=step, div=total_step,
            level=level, idx=idx)
        print(fname_8bit)
        tpg.img_wirte_float_as_16bit_int(fname_8bit, img_8bit)
        tpg.img_wirte_float_as_16bit_int(fname_10bit, img_10bit)


def grid_coordinate_test(
        width=1920, height=1080, h_num=6, v_num=4,
        fg_width=200, fg_height=150, remove_tblr_margin=False):
    gc = GridCoordinate(
        bg_width=width, bg_height=height,
        fg_width=fg_width, fg_height=fg_height,
        h_num=h_num, v_num=v_num, remove_tblr_margin=remove_tblr_margin)
    pos_list = gc.get_st_pos_list()

    img = np.zeros((height, width, 3))
    fg_img = np.ones((fg_height, fg_width, 3))
    for v_idx in range(v_num):
        for h_idx in range(h_num):
            idx = v_idx * h_num + h_idx
            tpg.merge(
                img, fg_img*(idx+1)/(h_num*v_num + 1), pos_list[h_idx][v_idx])
    tpg.img_wirte_float_as_16bit_int("./test_img.png", img)


def create_8bit_10bit_3x3_pattern(scale_factor=1):
    bg_width = 1920 * scale_factor
    bg_height = 1080 * scale_factor
    fg_width = 500 * scale_factor
    fg_height = 128 * scale_factor
    caption_font_size = 20 * scale_factor
    desc_font_size = 28 * scale_factor

    bg_img = np.ones((bg_height, bg_width, 3)) * 32 / 255
    tpg.draw_outline(bg_img, (0.5, 0.5, 0.5), 2)

    add_description_text(
        img=bg_img, img_width=bg_width, img_height=bg_height,
        font_size=desc_font_size)

    add_8bit_10bit_img(
        bg_img=bg_img, fg_width=fg_width, fg_height=fg_height,
        caption_font_size=caption_font_size)

    fname = f"./img/10bit_checker_{bg_width}x{bg_height}.png"
    tpg.img_wirte_float_as_16bit_int(fname, bg_img)


def add_8bit_10bit_img(
        bg_img, fg_width, fg_height, caption_font_size):
    bg_width = bg_img.shape[1]
    bg_height = bg_img.shape[0]
    block_width, block_height = calc_8bit_10bit_pair_block_size(
        width=fg_width, height=fg_height)
    dummy_text = "10bit"
    font_path = fc.NOTO_SANS_MONO_REGULAR

    level_list = [
        [tpg.L_LOW_C_LOW, tpg.L_MIDDLE_C_LOW, tpg.L_HIGH_C_LOW],
        [tpg.L_LOW_C_MIDDLE, tpg.L_MIDDLE_C_MIDDLE, tpg.L_HIGH_C_MIDDLE],
        [tpg.L_LOW_C_HIGH, tpg.L_MIDDLE_C_HIGH, tpg.L_HIGH_C_HIGH]]
    h_num, v_num = np.array(level_list).shape

    # internal of the block pos
    img_text_coord = ImgWithTextCoordinate(
        img_width=fg_width, img_height=fg_height,
        text=dummy_text, font_size=caption_font_size,
        text_pos="left", font_path=font_path,
        margin_num_of_chara=0.5)
    img_st_pos, text_st_pos = img_text_coord.get_img_and_text_st_pos()

    # calc block st pos
    gc = GridCoordinate(
        bg_width=bg_width, bg_height=bg_height,
        fg_width=block_width, fg_height=block_height,
        h_num=h_num, v_num=v_num, remove_tblr_margin=False)
    block_pos_list = gc.get_st_pos_list()

    for h_idx in range(h_num):
        for v_idx in range(v_num):
            dummy_img = np.zeros((block_height, block_width, 3))
            # tpg.draw_outline(dummy_img, (1, 0, 1), 1)
            # tpg.merge(bg_img, dummy_img, block_pos_list[h_idx][v_idx])
            add_each_8bit_10bit_img(
                img=bg_img, block_st_pos=block_pos_list[h_idx][v_idx],
                img_st_pos=img_st_pos, text_st_pos=text_st_pos,
                img_width=fg_width, img_height=fg_height,
                level=level_list[h_idx][v_idx], font_size=caption_font_size,
                font_path=font_path)


def add_each_8bit_10bit_img(
        img, block_st_pos, img_st_pos, text_st_pos, img_width, img_height,
        level, font_size, font_path):
    width = img_width - img_st_pos[0]
    img_out_8bit, img_out_10bit = tpg.create_8bit_10bit_id_patch(
        width=width, height=img_height, total_step=36,
        level=level, hdr10=False)
    # 8bit
    text_st = text_st_pos + block_st_pos
    img_st = img_st_pos + block_st_pos
    tpg.merge(img, img_out_8bit, img_st)
    text_drawer = fc.TextDrawer(
        img, text="8bit", pos=text_st,
        font_color=(0.5, 0.5, 0.5), font_size=font_size,
        font_path=font_path)
    text_drawer.draw()

    # 10bit
    text_st[1] += img_height + 2
    img_st[1] += img_height + 2
    tpg.merge(img, img_out_10bit, img_st)
    text_drawer = fc.TextDrawer(
        img, text="10bit", pos=text_st,
        font_color=(0.5, 0.5, 0.5), font_size=font_size,
        font_path=font_path)
    text_drawer.draw()


def add_description_text(img, img_width, img_height, font_size):
    text = f"Resolution: {img_width}x{img_height}, "
    text += "EOTF: Gamma 2.4, Gamut: BT.709, White point: D65"
    font_color = (0.5, 0.5, 0.5)
    font_path = fc.NOTO_SANS_MONO_REGULAR
    text_width, text_height = fc.get_text_width_height(
        text=text, font_path=font_path, font_size=font_size)
    margin = text_height // 4
    text_drawer = fc.TextDrawer(
        img, text=text, pos=(margin, img_height-text_height-margin),
        font_color=font_color, font_size=font_size,
        font_path=font_path)
    text_drawer.draw()

    text = f"Revision: {REVISION:02d}"
    text_width, text_height = fc.get_text_width_height(
        text=text, font_path=font_path, font_size=font_size)
    margin = text_height // 4
    text_drawer = fc.TextDrawer(
        img, text=text,
        pos=(img_width - margin - text_width, img_height-text_height-margin),
        font_color=font_color, font_size=font_size,
        font_path=font_path)
    text_drawer.draw()


def calc_8bit_10bit_pair_block_size(width, height):
    margin_between_8bit_10bit = 2
    block_width = width
    block_height = height * 2 + margin_between_8bit_10bit

    return block_width, block_height


def create_blog_8bit_10bit_image():
    width = 500
    height = 128
    st_level = 96 / 255
    ed_level = 216 / 255

    line = np.linspace(st_level, ed_level, width)
    line = np.dstack((line, line, line))
    img_base = line * np.ones((height, 1, 3))
    img_8bit = np.uint8(np.round(img_base * 255))

    img_6bit = img_8bit & 0xFC

    tpg.img_write("./img/blog_8bit.png", img_8bit)
    tpg.img_write("./img/blog_6bit.png", img_6bit)


def main_func():
    # create_and_save_8bit_10bit_patch(
    #     width=512, height=512, total_step=20, direction='h', level='low')
    # create_and_save_8bit_10bit_patch(
    #     width=512, height=512, total_step=20, direction='v', level='low')
    # create_and_save_8bit_10bit_patch(
    #     width=512, height=512, total_step=20, direction='h', level='middle')
    # create_and_save_8bit_10bit_patch(
    #     width=512, height=512, total_step=20, direction='v', level='middle')
    # create_and_save_8bit_10bit_patch(
    #     width=512, height=512, total_step=20, direction='h', level='high')
    # create_and_save_8bit_10bit_patch(
    #     width=512, height=512, total_step=20, direction='v', level='high')
    # id_patch_generator_class_test(
    #     width=512, height=512, total_step=20, level=tpg.L_HIGH_C_HIGH, step=4)
    # grid_coordinate_test()
    # grid_coordinate_test(
    #     width=200, height=1080, h_num=1, v_num=3,
    #     fg_width=200, fg_height=150, remove_tblr_margin=True)
    # create_8bit_10bit_3x3_pattern(scale_factor=1)
    # create_8bit_10bit_3x3_pattern(scale_factor=2)
    create_blog_8bit_10bit_image()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
