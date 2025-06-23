# -*- coding: utf-8 -*-

# import standard libraries
import os

# import third-party libraries
from test_pattern_generator2 import png_to_avif_2

# import my libraries
import color_space as cs
import transfer_functions as tf

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    png_to_avif_2(
        png_fname="./src_img/1920x1080_sRGB_sRGB.png",
        avif_fname="./src_img/1920x1080_sRGB_sRGB_fixed_cll.avif",
        color_space_name=cs.BT709,
        transfer_characteristics=tf.SRGB,
        cll=10000, pall=10000
    )
    png_to_avif_2(
        png_fname="./src_img/1920x1080_sRGB_sRGB.png",
        avif_fname="./src_img/1920x1080_sRGB_sRGB.avif",
        color_space_name=cs.BT709,
        transfer_characteristics=tf.SRGB,
        # cll=10000, pall=10000
    )

    png_to_avif_2(
        png_fname="./src_img/1920x1080_sRGB_sRGB_0.5x.png",
        avif_fname="./src_img/1920x1080_sRGB_sRGB_0.5x_fixed_cll.avif",
        color_space_name=cs.BT709,
        transfer_characteristics=tf.SRGB,
        cll=10000, pall=10000
    )
    png_to_avif_2(
        png_fname="./src_img/1920x1080_sRGB_sRGB_0.5x.png",
        avif_fname="./src_img/1920x1080_sRGB_sRGB_0.5x.avif",
        color_space_name=cs.BT709,
        transfer_characteristics=tf.SRGB,
        # cll=10000, pall=10000
    )

    png_to_avif_2(
        png_fname="./src_img/1920x1080_ST2084_Rec.2020.png",
        avif_fname="./src_img/1920x1080_ST2084_Rec.2020_fixed_cll.avif",
        color_space_name=cs.BT2020,
        transfer_characteristics=tf.ST2084,
        cll=10000, pall=10000
    )
    png_to_avif_2(
        png_fname="./src_img/1920x1080_ST2084_Rec.2020.png",
        avif_fname="./src_img/1920x1080_ST2084_Rec.2020.avif",
        color_space_name=cs.BT2020,
        transfer_characteristics=tf.ST2084,
        # cll=10000, pall=10000
    )
