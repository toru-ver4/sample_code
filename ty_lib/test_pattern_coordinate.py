#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
coordinate calculation
"""

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg
import font_control as fc


class GridCoordinate():
    def __init__(
            self, bg_width=1920, bg_height=1080,
            fg_width=200, fg_height=100, h_num=5, v_num=4,
            remove_tblr_margin=False):
        """
        Example
        -------
        >>> gc = GridCoordinate(
        ...     bg_width=1920, bg_height=1080,
        ...     fg_width=200, fg_height=150,
        ...     h_num=3, v_num=2, remove_tblr_margin=False)
        >>> pos_list = gc.get_st_pos_list()
        [[[ 330  260] [ 330  670]]
         [[ 860  260] [ 860  670]]
         [[1390  260] [1390  670]]]

        >>> img = np.zeros((height, width, 3))
        >>> fg_img = np.ones((fg_height, fg_width, 3))
        >>> for v_idx in range(v_num):
        ...     for h_idx in range(h_num):
        ...         idx = v_idx * h_num + h_idx
        ...         tpg.merge(img, fg_img, pos_list[h_idx][v_idx])
        >>> tpg.img_wirte_float_as_16bit_int("./test_img.png", img)
        """
        self.remove_tblr_margin = remove_tblr_margin

        if (fg_width*h_num > bg_width) or (fg_height*v_num > bg_height):
            print("Fatal Error!")
            print("  fg_width or fg_height is too big.")

        if self.remove_tblr_margin:
            self.tblr_maring_offset = -1
        else:
            self.tblr_maring_offset = 1

        h_margin_list, v_margin_list = self.calc_hv_margin_list(
            bg_width=bg_width, bg_height=bg_height,
            fg_width=fg_width, fg_height=fg_height, h_num=h_num, v_num=v_num)
        self.calc_st_coordinate_list(
            h_num=h_num, v_num=v_num, fg_width=fg_width, fg_height=fg_height,
            h_margin_list=h_margin_list, v_margin_list=v_margin_list)

    def get_st_pos_list(self):
        return self.pos_list

    def calc_hv_margin_list(
            self, bg_width=1920, bg_height=1080,
            fg_width=200, fg_height=100, h_num=5, v_num=4):
        h_margin_total = bg_width - (fg_width * h_num)
        v_margin_total = bg_height - (fg_height * v_num)
        h_margin_list = tpg.equal_devision(
            h_margin_total, h_num + self.tblr_maring_offset)
        v_margin_list = tpg.equal_devision(
            v_margin_total, v_num + self.tblr_maring_offset)

        return h_margin_list, v_margin_list

    def calc_st_coordinate_list(
            self, h_num, v_num, fg_width, fg_height,
            h_margin_list, v_margin_list):
        """
        calculate start position.

        Example
        -------
        h_num : int
            horizontal number of the fg_pattern.
        v_num : int
            vertical number of the fg_pattern.
        fg_width : int
            width of the fg pattern.
        fg_height : int
            height of the fg pattern.
        h_margin_list : list
            horizontal margin list
        v_margin_list : list
            vertical margin list

        Returns
        -------
        self.pos_list (not return, just keep on the ram) : ndarray
            pos_list[h_idx][v_idx][0] : h_pos
            pos_list[h_idx][v_idx][1] : v_pos
        """
        if self.remove_tblr_margin:
            st_offset_h = 0
            st_offset_v = 0
        else:
            st_offset_h = h_margin_list.pop(0)
            st_offset_v = v_margin_list.pop(0)

        self.pos_list = np.zeros(
            (h_num, v_num, 2), dtype=np.uint32)
        v_pos = st_offset_v
        for v_idx in range(v_num):
            h_pos = st_offset_h
            for h_idx in range(h_num):
                self.pos_list[h_idx][v_idx][0] = h_pos
                self.pos_list[h_idx][v_idx][1] = v_pos
                if h_idx >= (h_num - 1):
                    break
                h_magin = h_margin_list[h_idx]
                h_pos += (h_magin + fg_width)
            if v_idx >= (v_num - 1):
                break
            v_margin = v_margin_list[v_idx]
            v_pos += (v_margin + fg_height)


class ImgWithTextCoordinate():
    def __init__(
            self, img_width, img_height,
            text="8bit \nwith alpha", font_size=30,
            text_pos="left",
            font_path=fc.NOTO_SANS_MONO_BOLD,
            margin_num_of_chara=0.5):
        """
        Parameters
        ----------
        img_width : int
            width of the image
        img_height : int
            haight of the image
        text_pos : string
            "left", "right", "top", "bottom".
        margin_num_of_chara : float
            the margin rate between text and image.
            1.0 means one character width.

        Examples
        --------
        >>> text = "8bit \nwith alpha"
        >>> font_size = 30
        >>> font_path = fc.NOTO_SANS_MONO_REGULAR
        >>> margin_num_of_chara = 1.0
        >>> fg_img = np.ones((300, 400, 3)) * np.array([0, 1, 1])
        >>> # left
        >>> bg_img = np.zeros((720, 1280, 3))
        >>> tpg.draw_outline(bg_img, np.array([0, 1, 0]), 1)
        >>> img_text_coorinate = ImgWithTextCoordinate(
        ...     img_width=fg_img.shape[1], img_height=fg_img.shape[0]
        ...     text=text, font_size=font_size,
        ...     text_pos="left", font_path=font_path,
        ...     margin_num_of_chara=margin_num_of_chara)
        >>> img_st_pos, text_st_pos\
        ...     = img_text_coorinate.get_img_and_text_st_pos()
        >>> tpg.merge(bg_img, fg_img, img_st_pos)
        >>> text_drawer = fc.TextDrawer(
        ...     bg_img, text=text, pos=text_st_pos,
        ...     font_color=(0.5, 0.5, 0.5), font_size=font_size,
        ...     font_path=font_path)
        >>> text_drawer.draw()
        >>> tpg.img_wirte_float_as_16bit_int("./img_left.png", bg_img)
        """
        text_width, text_height = self.calc_text_size(
            text=text, font_size=font_size, font_path=font_path)

        text_img_margin = self.calc_text_img_margin(
            text=text, text_width=text_width,
            margin_num_of_chara=margin_num_of_chara)

        if text_pos == "left":
            text_st_pos = (0, 0)
            img_st_pos = (text_width + text_img_margin, 0)
        elif text_pos == "right":
            text_st_pos = (img_width + text_img_margin, 0)
            img_st_pos = (0, 0)
        elif text_pos == "top":
            text_st_pos = (0, 0)
            img_st_pos = (0, text_height + text_img_margin)
        elif text_pos == "bottom":
            text_st_pos = (0, img_height + text_img_margin)
            img_st_pos = (0, 0)
        else:
            print("Parameter error!")
            print(f"  text_pos={text_pos} is invalid")

        self.text_st_pos = np.array(text_st_pos, dtype=np.uint32)
        self.img_st_pos = np.array(img_st_pos, dtype=np.uint32)

    def get_img_and_text_st_pos(self):
        return self.img_st_pos, self.text_st_pos

    def calc_text_size(self, text, font_size, font_path):
        text_drawer = fc.TextDrawer(
            None, text=text, font_size=font_size,
            font_path=font_path)
        text_drawer.make_text_img_with_alpha()
        text_width, text_height = text_drawer.get_text_size()

        return text_width, text_height

    def calc_text_img_margin(
            self, text, text_width, margin_num_of_chara=0.5):
        """
        Parameters
        ----------
        test : str
            text
        text_width : int
            text width. unit is pixel
        margin_num_of_chara : float
            the margin rate between text and image.
            1.0 means one character width.
        """
        str_length = self.get_text_horizontal_length(text)
        text_width_one_str = text_width / str_length * margin_num_of_chara

        return int(text_width_one_str + 0.5)

    def get_text_horizontal_length(self, text):
        str_list_splitted_lf = text.split('\n')
        str_length_num_list = [
            len(strings) for strings in str_list_splitted_lf]
        max_length = np.array(str_length_num_list).max()

        return max_length


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # grid coordinate
    gc = GridCoordinate(
        bg_width=1920, bg_height=1080,
        fg_width=200, fg_height=150,
        h_num=3, v_num=2, remove_tblr_margin=False)
    pos_list = gc.get_st_pos_list()

    text = "8bit \nwith alpha"
    font_size = 30
    font_path = fc.NOTO_SANS_MONO_REGULAR
    margin_num_of_chara = 1.0
    fg_img = np.ones((300, 400, 3)) * np.array([0, 1, 1])

    # left
    bg_img = np.zeros((720, 1280, 3))
    tpg.draw_outline(bg_img, np.array([0, 1, 0]), 1)
    img_text_coorinate = ImgWithTextCoordinate(
        img_width=fg_img.shape[1], img_height=fg_img.shape[0],
        text=text, font_size=font_size,
        text_pos="left", font_path=font_path,
        margin_num_of_chara=margin_num_of_chara)
    img_st_pos, text_st_pos = img_text_coorinate.get_img_and_text_st_pos()
    tpg.merge(bg_img, fg_img, img_st_pos)
    text_drawer = fc.TextDrawer(
        bg_img, text=text, pos=text_st_pos,
        font_color=(0.5, 0.5, 0.5), font_size=font_size,
        font_path=font_path)
    text_drawer.draw()
    tpg.img_wirte_float_as_16bit_int("./img_left.png", bg_img)

    # right
    bg_img = np.zeros((720, 1280, 3))
    tpg.draw_outline(bg_img, np.array([0, 1, 0]), 1)
    img_text_coorinate = ImgWithTextCoordinate(
        img_width=fg_img.shape[1], img_height=fg_img.shape[0],
        text=text, font_size=font_size,
        text_pos="right", font_path=font_path,
        margin_num_of_chara=margin_num_of_chara)
    img_st_pos, text_st_pos = img_text_coorinate.get_img_and_text_st_pos()
    tpg.merge(bg_img, fg_img, img_st_pos)
    text_drawer = fc.TextDrawer(
        bg_img, text=text, pos=text_st_pos,
        font_color=(0.5, 0.5, 0.5), font_size=font_size,
        font_path=font_path)
    text_drawer.draw()
    tpg.img_wirte_float_as_16bit_int("./img_right.png", bg_img)

    # top
    bg_img = np.zeros((720, 1280, 3))
    tpg.draw_outline(bg_img, np.array([0, 1, 0]), 1)
    img_text_coorinate = ImgWithTextCoordinate(
        img_width=fg_img.shape[1], img_height=fg_img.shape[0],
        text=text, font_size=font_size,
        text_pos="top", font_path=font_path,
        margin_num_of_chara=margin_num_of_chara)
    img_st_pos, text_st_pos = img_text_coorinate.get_img_and_text_st_pos()
    tpg.merge(bg_img, fg_img, img_st_pos)
    text_drawer = fc.TextDrawer(
        bg_img, text=text, pos=text_st_pos,
        font_color=(0.5, 0.5, 0.5), font_size=font_size,
        font_path=font_path)
    text_drawer.draw()
    tpg.img_wirte_float_as_16bit_int("./img_top.png", bg_img)

    # bottom
    bg_img = np.zeros((720, 1280, 3))
    tpg.draw_outline(bg_img, np.array([0, 1, 0]), 1)
    img_text_coorinate = ImgWithTextCoordinate(
        img_width=fg_img.shape[1], img_height=fg_img.shape[0],
        text=text, font_size=font_size,
        text_pos="bottom", font_path=font_path,
        margin_num_of_chara=margin_num_of_chara)
    img_st_pos, text_st_pos = img_text_coorinate.get_img_and_text_st_pos()
    tpg.merge(bg_img, fg_img, img_st_pos)
    text_drawer = fc.TextDrawer(
        bg_img, text=text, pos=text_st_pos,
        font_color=(0.5, 0.5, 0.5), font_size=font_size,
        font_path=font_path)
    text_drawer.draw()
    tpg.img_wirte_float_as_16bit_int("./img_bottom.png", bg_img)
