# -*- coding: utf-8 -*-

# import standard libraries
import os

# import third-party libraries
from test_pattern_generator2 import png_to_avif_2
import numpy as np

# import my libraries
import color_space as cs
import transfer_functions as tf
import test_pattern_generator2 as tpg

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # png_to_avif_2(
    #     png_fname="./src_img/1920x1080_sRGB_sRGB.png",
    #     avif_fname="./src_img/1920x1080_sRGB_sRGB_fixed_cll.avif",
    #     color_space_name=cs.BT709,
    #     transfer_characteristics=tf.SRGB,
    #     cll=10000, pall=10000
    # )
    # png_to_avif_2(
    #     png_fname="./src_img/1920x1080_sRGB_sRGB.png",
    #     avif_fname="./src_img/1920x1080_sRGB_sRGB.avif",
    #     color_space_name=cs.BT709,
    #     transfer_characteristics=tf.SRGB,
    #     # cll=10000, pall=10000
    # )

    # png_to_avif_2(
    #     png_fname="./src_img/1920x1080_sRGB_sRGB_0.5x.png",
    #     avif_fname="./src_img/1920x1080_sRGB_sRGB_0.5x_fixed_cll.avif",
    #     color_space_name=cs.BT709,
    #     transfer_characteristics=tf.SRGB,
    #     cll=10000, pall=10000
    # )
    # png_to_avif_2(
    #     png_fname="./src_img/1920x1080_sRGB_sRGB_0.5x.png",
    #     avif_fname="./src_img/1920x1080_sRGB_sRGB_0.5x.avif",
    #     color_space_name=cs.BT709,
    #     transfer_characteristics=tf.SRGB,
    #     # cll=10000, pall=10000
    # )

    # png_to_avif_2(
    #     png_fname="./src_img/1920x1080_ST2084_Rec.2020.png",
    #     avif_fname="./src_img/1920x1080_ST2084_Rec.2020_fixed_cll.avif",
    #     color_space_name=cs.BT2020,
    #     transfer_characteristics=tf.ST2084,
    #     cll=10000, pall=10000
    # )
    # png_to_avif_2(
    #     png_fname="./src_img/1920x1080_ST2084_Rec.2020.png",
    #     avif_fname="./src_img/1920x1080_ST2084_Rec.2020.avif",
    #     color_space_name=cs.BT2020,
    #     transfer_characteristics=tf.ST2084,
    #     # cll=10000, pall=10000
    # )

    # png_to_avif_2(
    #     png_fname="./src_img/kanazawa_01_BT2100-PQ.png",
    #     avif_fname="./src_img/kanazawa_01_BT2100-PQ.avif",
    #     color_space_name=cs.BT2020,
    #     transfer_characteristics=tf.ST2084,
    # )
    # png_to_avif_2(
    #     png_fname="./src_img/kanazawa_01_sRGB.png",
    #     avif_fname="./src_img/kanazawa_01_sRGB.avif",
    #     color_space_name=cs.BT709,
    #     transfer_characteristics=tf.SRGB,
    # )

    # png_to_avif_2(
    #     png_fname="./src_img/kanazawa_castle_01_BT2100-PQ.png",
    #     avif_fname="./src_img/kanazawa_castle_01_BT2100-PQ.avif",
    #     color_space_name=cs.BT2020,
    #     transfer_characteristics=tf.ST2084,
    # )
    # png_to_avif_2(
    #     png_fname="./src_img/kanazawa_castle_01_sRGB.png",
    #     avif_fname="./src_img/kanazawa_castle_01_sRGB.avif",
    #     color_space_name=cs.BT709,
    #     transfer_characteristics=tf.SRGB,
    # )

    # png_to_avif_2(
    #     png_fname="./src_img/kenrokuen_01_BT2100-PQ.png",
    #     avif_fname="./src_img/kenrokuen_01_BT2100-PQ.avif",
    #     color_space_name=cs.BT2020,
    #     transfer_characteristics=tf.ST2084,
    # )
    img = tpg.img_read_as_float("./src_img/kenrokuen_01_BT2100-PQ.png")
    img = np.clip(img, 0.0, 769/1023)
    tpg.img_wirte_float_as_16bit_int("./src_img/kenrokuen_01_BT2100-PQ_1000nit.png", img)
    png_to_avif_2(
        png_fname="./src_img/kenrokuen_01_BT2100-PQ_1000nit.png",
        avif_fname="./src_img/kenrokuen_01_BT2100-PQ_1000nit.avif",
        color_space_name=cs.BT2020,
        transfer_characteristics=tf.ST2084,
    )
    img = tpg.img_read_as_float("./src_img/kenrokuen_01_BT2100-PQ.png")
    img = np.clip(img, 0.0, 1022/1023)
    tpg.img_wirte_float_as_16bit_int("./src_img/kenrokuen_01_BT2100-PQ_10000nit.png", img)
    png_to_avif_2(
        png_fname="./src_img/kenrokuen_01_BT2100-PQ_10000nit.png",
        avif_fname="./src_img/kenrokuen_01_BT2100-PQ_10000nit.avif",
        color_space_name=cs.BT2020,
        transfer_characteristics=tf.ST2084,
    )
    # png_to_avif_2(
    #     png_fname="./src_img/kenrokuen_01_sRGB.png",
    #     avif_fname="./src_img/kenrokuen_01_sRGB.avif",
    #     color_space_name=cs.BT709,
    #     transfer_characteristics=tf.SRGB,
    # )

    # hdr_capacity_list = [
    #     0.000, 0.563, 0.978, 1.300, 1.563, 1.978, 2.300, 2.563, 2.885, 3.300, 3.622, 3.885, 4.300, 5.622, 6.965,
    # ]
    # for hdr_capacity in hdr_capacity_list:
    #     png_to_avif_2(
    #         png_fname=f"./src_img/HDR_Capacity_{hdr_capacity:.3f}_1280x720.png",
    #         avif_fname=f"./src_img/HDR_Capacity_{hdr_capacity:.3f}_1280x720.avif",
    #         color_space_name=cs.BT2020,
    #         transfer_characteristics=tf.ST2084
    #     )
    # png_to_avif_2(
    #     png_fname="./src_img/HDR_Capacity_SDR_1280x720.png",
    #     avif_fname="./src_img/HDR_Capacity_SDR_1280x720.avif",
    #     color_space_name=cs.BT2020,
    #     transfer_characteristics=tf.SRGB,
    # )
