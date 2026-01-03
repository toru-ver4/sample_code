from pathlib import Path
import numpy as np
import os
import sys

from colour.models import RGB_COLOURSPACE_BT2020
from colour.io import write_image, read_image
from imagecodecs import JPEGXR, imread

import test_pattern_generator2 as tpg
import transfer_functions as tf
from create_ultrahdr_image import OFFSET_VAL, linearize_input_image
import color_space as cs

tp_module_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../03_Investigate_Full_Range_Encoding/"))
sys.path.insert(0, tp_module_path)

from create_src_test_pattern import (
    calc_gradation_pattern_block_st_pos,
    calc_rgbmyc_pattern_block_st_pos,
    calc_color_checker_pattern_block_st_pos
)


def create_hdr_from_sdr():
    gain_map_img_fname = "./gain_map_img/gain_map_HDR_Capacity_2.300_1280x720-HDR_Capacity_SDR_1280x720.jpeg"
    sdr_img_fname = "./src_png/HDR_Capacity_SDR_1280x720_8bit.jpeg"

    gain_map = tpg.img_read_as_float(gain_map_img_fname)
    sdr_linear = linearize_input_image(sdr_img_fname, tf_name=tf.SRGB, cs_name=cs.BT2020)
    sdr_linear = sdr_linear * 203 / 100
    gain_map_min = np.array([0.004646, 0.004646, 0.004646])
    gain_map_max = np.array([3.212598, 3.767513, 3.767513])

    gain_map_2 = np.zeros_like(gain_map)
    for idx in range(3):
        gain_map_2[..., idx] = (gain_map_max[idx] - gain_map_min[idx]) * gain_map[..., idx] + gain_map_min[idx]

    min_val = np.min(gain_map_2, axis=(0, 1))
    max_val = np.max(gain_map_2, axis=(0, 1))
    print(min_val, max_val)

    # gain_map_raw = np.log2((hdr_linear + OFFSET_VAL)/(sdr_linear + OFFSET_VAL))
    gain = 2 ** gain_map_2
    hdr_img = ((sdr_linear + OFFSET_VAL) * gain) - OFFSET_VAL
    # hdr_img = ((sdr_linear + OFFSET_VAL2) * gain) - OFFSET_VAL2
    hdr_img_nits = hdr_img * 100
    hdr_img_pq = tf.oetf_from_luminance(hdr_img_nits, tf.ST2084)

    print(np.max(hdr_img_nits))

    tpg.img_wirte_float_as_16bit_int("./debug_pq.png", hdr_img_pq)


def convert_png_to_avif():
    tpg.png_to_avif_2(
        png_fname="./src_png/HDR_Capacity_2.300_1280x720.png",
        avif_fname="./src_png/HDR_Capacity_2.300_1280x720.avif",
        lossless=True,
        cll=10000, pall=10000
    )


def compare_png_jpeg(jpeg_fname, png_fname):
    img_jpeg = tpg.img_read_as_float(jpeg_fname)
    img_png = tpg.img_read_as_float(png_fname)
    img_png = (np.round(img_png * 255)) / 255

    diff = img_jpeg - img_png
    ave = np.mean(diff)
    std = np.std(diff)
    max = np.max(diff)
    min = np.min(diff)

    print(f"ave={ave}, std={std}, min={min*255}, max={max*255}")


def create_10bit_pattern_for_full_range_encode():
    output_fname_png = "./src_png/lut_check_pattern_hdr.png"

    width = 1920
    height = 1080
    cv_max = 1023
    grey_patch_size = 32
    color_patch_size = 64
    img = np.zeros((height, width, 3), dtype=np.uint16)
    grey_block_img_base = np.ones((grey_patch_size, grey_patch_size, 3), dtype=np.uint16)
    color_block_img_base = np.ones((color_patch_size, color_patch_size, 3), dtype=np.uint16)

    for cv in range(cv_max+1):
        block_img = grey_block_img_base * cv
        st_pos = calc_gradation_pattern_block_st_pos(
            code_value=cv, width=width, block_size=grey_patch_size
        )
        tpg.merge(img, block_img, st_pos)

    # RGBMYC
    color_list = np.array(
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 0], [0, 1.0, 1.0]],
    )
    color_list_cv = tf.oetf_from_luminance(color_list * 100, tf.ST2084)
    color_list_10bit = np.round(color_list_cv * cv_max).astype(np.uint16)
    for color_idx in range(len(color_list_10bit)):
        block_img = color_block_img_base * color_list_10bit[color_idx]
        st_pos = calc_rgbmyc_pattern_block_st_pos(
            color_idx=color_idx, width=width,
            grey_block_size=grey_patch_size, color_block_size=color_patch_size
        )
        print(st_pos)
        tpg.merge(img, block_img, st_pos)

    # Color Checker
    color_checker_linear = tpg.generate_color_checker_rgb_value(color_space=RGB_COLOURSPACE_BT2020)
    rgb_value_pq = tf.oetf_from_luminance(color_checker_linear * 100, tf.ST2084)
    rgb_value_10bit = np.uint16(np.round(rgb_value_pq * cv_max))

    for color_idx in range(len(rgb_value_10bit)):
        block_img = color_block_img_base * rgb_value_10bit[color_idx]
        st_pos = calc_color_checker_pattern_block_st_pos(
            color_idx=color_idx, width=width,
            grey_block_size=grey_patch_size, color_block_size=color_patch_size
        )
        tpg.merge(img, block_img, st_pos)

    tpg.img_wirte_float_as_16bit_int(
        filename=output_fname_png, img_float=img/255
    )

    sdr_fname = "./src_png/lut_check_pattern_sdr.png"
    img = np.ones((height, width, 3)) * 0.5
    tpg.img_wirte_float_as_16bit_int(filename=sdr_fname, img_float=img)


def extract_tp_data_from_screenshot():
    src_fname = "./debug/tp_capture_hdr.jxr"
    dst_fname = "./debug/tp_capture.exr"

    capture_img = imread(src_fname)
    height, width = capture_img.shape[:2]
    st_v = (height // 2) - (height // 4)
    ed_v = (height // 2) + (height // 4)

    st_h = (width // 2) - (width // 4)
    ed_h = (width // 2) + (width // 4)
    
    dst_img = capture_img[st_v:ed_v, st_h:ed_h]
    dst_img[..., :3] = dst_img[..., :3] * 0.8
    
    write_image(image=dst_img, path=dst_fname, bit_depth='float16')


def analyze_decoded_ultrahdr():
    decoded_image_file = "./debug/tp_capture.exr"
    decoded_img = read_image(path=decoded_image_file)

    ramp_pos_x = np.round(np.linspace(62, 1854, 65)).astype(np.uint16)
    ramp_pos_y = 845
    measured_value = decoded_img[ramp_pos_y, ramp_pos_x, 1]
    print(measured_value)

    # reference
    gain_map_img_fname = "./gain_map_img/gain_map_1920x1080_ST2084_Rec.2020-lut_check_pattern_sdr.jpeg"
    sdr_img_fname = "./src_png/lut_check_pattern_sdr.png"

    gain_map = tpg.img_read_as_float(gain_map_img_fname)
    sdr_linear = linearize_input_image(sdr_img_fname, tf_name=tf.SRGB, cs_name=cs.BT2020)
    sdr_linear = sdr_linear * 203 / 100
    gain_map_min = np.array([-5.83514575, -5.83514575, -5.83514575])
    gain_map_max = np.array([7.80882315, 7.80882315, 7.80882315])

    gain_map_2 = np.zeros_like(gain_map)
    for idx in range(3):
        gain_map_2[..., idx] = (gain_map_max[idx] - gain_map_min[idx]) * gain_map[..., idx] + gain_map_min[idx]

    min_val = np.min(gain_map_2, axis=(0, 1))
    max_val = np.max(gain_map_2, axis=(0, 1))

    # gain_map_raw = np.log2((hdr_linear + OFFSET_VAL)/(sdr_linear + OFFSET_VAL))
    gain = 2 ** gain_map_2
    hdr_img = ((sdr_linear + OFFSET_VAL) * gain) - OFFSET_VAL
    ref_value = hdr_img[ramp_pos_y, ramp_pos_x, 1]
    print(ref_value)
    diff = np.abs(measured_value - ref_value)
    diff_rate = diff / ref_value
    # print(diff_rate)
    # print(diff)


def output_luminance(p_int, cv: list):
    cv = np.array(cv)
    cv_ave = np.round(np.mean(cv))

    luminance = tf.eotf_to_luminance(cv_ave/1023, tf.ST2084)
    print(f"{p_int}/992, {cv_ave}/1023, {luminance:.1f} nits")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_hdr_from_sdr()
    # convert_png_to_avif()
    # compare_png_jpeg(
    #     jpeg_fname="./src_png/HDR_Capacity_SDR_1280x720_8bit.jpeg",
    #     png_fname="./src_png/HDR_Capacity_SDR_1280x720.png",
    # )

    # create_10bit_pattern_for_full_range_encode()
    # extract_tp_data_from_screenshot()
    # analyze_decoded_ultrahdr()

    # switch 2
    # cv496 = 496/1023
    # cv499 = 499/1023
    # cv511 = 511/1023
    # cv522 = 522/1023
    # cv531 = 531/1023
    # cv641 = 641/1023
    # cv645 = 645/1023
    # cv712 = 712/1023
    # cv714 = 714/1023
    # data = np.array([cv496, cv499, cv511, cv522, cv531, cv641, cv645, cv712, cv714])
    # print(tf.eotf_to_luminance(data, tf.ST2084))
    # print(np.round(tf.oetf_from_luminance(np.array([80, 100, 1000]), tf.ST2084) * 1023).astype(np.uint32))

    # p1_000 = [496, 499, 496]
    # p1_001 = [509, 511, 508]
    # p1_002 = [519, 522, 519]
    # p1_003 = [530, 532, 529]
    # p1_004 = [539, 541, 538]
    # p1_005 = [546, 548, 545]
    # p1_006 = [554, 557, 554]
    # p1_052 = [712, 712, 710]
    # p1_092 = [769, 772, 769]
    # p1_992 = [1021, 1021, 1021]

    # output_luminance(0, p1_000)
    # output_luminance(1, p1_001)
    # output_luminance(2, p1_002)
    # output_luminance(3, p1_003)
    # output_luminance(4, p1_004)
    # output_luminance(5, p1_005)
    # output_luminance(6, p1_006)
    # output_luminance(52, p1_052)
    # output_luminance(92, p1_092)
    # output_luminance(992, p1_992)
