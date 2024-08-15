import cv2
from pathlib import Path
import numpy as np
import os

from PIL import Image

import test_pattern_generator2 as tpg
import transfer_functions as tf
import color_space as cs

GAIN_MAP_CS_NAME = cs.BT2020
SDR_WHITE_LUMINANCE = 203
OFFSET_VAL = 1/128  # k_sdr, k_hdr parameter in Adobe specification


def linearize_input_image(fname, tf_name=tf.ST2084, cs_name=cs.P3_D65):
    img = tpg.img_read_as_float(fname)
    linear_img = tf.eotf_to_luminance(img, tf_name) / tf.REF_WHITE_LUMINANCE
    large_xyz = cs.rgb_to_large_xyz(
        rgb=linear_img, color_space_name=cs_name)
    linear_img = cs.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=GAIN_MAP_CS_NAME)

    return linear_img


def img_write_8bit_jpeg_from_float(filename: str, img_float: np.ndarray):
    img = np.round(img_float * 0xFF).astype(np.uint8)
    img_pil = Image.fromarray(img)
    with open("./icc_profile/sRGB_BT2020.icc", 'rb') as f:
        icc_profile = f.read()
    img_pil.save(
        filename, 'JPEG', quality=100, icc_profile=icc_profile, subsampling=0
    )


def png_16bit_to_rgba1010102(fname: str):

    img = tpg.img_read(fname)
    height, width = img.shape[:2]
    alpha_channel = np.full((height, width), 0xFFFF, dtype=np.uint16)
    rgba1010102 = np.zeros((width, height), dtype=np.uint32)

    rgba_image = np.dstack((img, alpha_channel))

    r = (rgba_image[:, :, 0] >> 6).astype(np.uint32)
    g = (rgba_image[:, :, 1] >> 6).astype(np.uint32)
    b = (rgba_image[:, :, 2] >> 6).astype(np.uint32)
    a = (rgba_image[:, :, 3] >> 14).astype(np.uint32)

    rgba1010102 = r | (g << 10) | (b << 20) | (a << 30)

    out_fname = fname.replace(".png", "_rgba1010102.raw")
    print(out_fname)
    rgba1010102.tofile(out_fname)


def png_16bit_to_rgba8888(fname: str):

    img = tpg.img_read(fname)
    height, width = img.shape[:2]
    alpha_channel = np.full((height, width), 0xFFFF, dtype=np.uint16)
    rgba8888 = np.zeros((width, height), dtype=np.uint32)

    rgba_image = np.dstack((img, alpha_channel))

    r = (rgba_image[:, :, 0] >> 8).astype(np.uint32)
    g = (rgba_image[:, :, 1] >> 8).astype(np.uint32)
    b = (rgba_image[:, :, 2] >> 8).astype(np.uint32)
    a = (rgba_image[:, :, 3] >> 8).astype(np.uint32)

    rgba8888 = r | (g << 8) | (b << 16) | (a << 24)
    # print(np.vectorize(hex)(rgba8888))

    out_fname = fname.replace(".png", "_rgba8888.raw")
    print(out_fname)
    rgba8888.tofile(out_fname)


def create_test_data():
    img = np.arange((27)).astype(np.uint16).reshape(3, 3, 3)
    img = img << 6
    tpg.img_write("./test_data.png", img)


def _debug_calc_gain_map_metadata(hdr_fname, sdr_fname):
    img_hdr = tpg.img_read_as_float(filename=hdr_fname)
    img_sdr = tpg.img_read_as_float(filename=sdr_fname)

    cfg_name = f"./metadata_{Path(hdr_fname).stem}-{Path(sdr_fname).stem}.cfg"
    print(cfg_name)

    kk = 0.00001
    gg = np.log2((img_hdr + kk)/(img_sdr + kk))

    with open(cfg_name, 'wt') as f:
        buf = ""
        buf += f"--maxContentBoost {np.max(gg):.3f}\n"
        buf += f"--minContentBoost {np.min(gg):.3f}\n"
        buf += "--gamma 1.0\n"
        buf += "--offsetSdr 0.0\n"
        buf += "--offsetHdr 0.0\n"
        buf += "--hdrCapacityMin 1.0\n"
        buf += "--hdrCapacityMax 2.3\n"
        f.write(buf)


def save_gain_map_metadata(
        hdr_fname, sdr_fname, min_val, max_val, offset_val,
        hdr_capacity_min=0.0, hdr_capacity_max=2.3):
    cfg_name = f"./metadata/metadata_{Path(hdr_fname).stem}-"
    cfg_name += f"{Path(sdr_fname).stem}.cfg"
    print(cfg_name)

    with open(cfg_name, 'wt') as f:
        buf = ""
        buf += f"--maxContentBoost {2**max_val}\n"
        buf += f"--minContentBoost {2**min_val}\n"
        buf += "--gamma 1.0\n"
        buf += f"--offsetSdr {offset_val}\n"
        buf += f"--offsetHdr {offset_val}\n"
        buf += f"--hdrCapacityMin {hdr_capacity_min}\n"
        buf += f"--hdrCapacityMax {hdr_capacity_max}\n"
        f.write(buf)


def make_sdr_8bit_jpeg(sdr_fname: str):
    img = tpg.img_read_as_float(sdr_fname)
    fname = sdr_fname.replace(".png", "_8bit.jpeg")
    print(fname)
    img_write_8bit_jpeg_from_float(filename=fname, img_float=img)


def create_gain_map_jpeg_and_metadata(hdr_fname, sdr_fname, hdr_tf):
    sdr_linear = linearize_input_image(
        fname=sdr_fname, tf_name=tf.SRGB, cs_name=cs.BT2020
    )
    sdr_linear = sdr_linear * SDR_WHITE_LUMINANCE / tf.REF_WHITE_LUMINANCE

    hdr_linear = linearize_input_image(
        fname=hdr_fname, tf_name=hdr_tf, cs_name=cs.BT2020
    )

    gain_map_raw = np.log2((hdr_linear + OFFSET_VAL)/(sdr_linear + OFFSET_VAL))

    min_val = np.min(gain_map_raw)
    max_val = np.max(gain_map_raw)
    gain_map_normalized = (gain_map_raw - min_val) / (max_val - min_val)

    gain_map_fname = "./gain_map_img/gain_map_"
    gain_map_fname += f"{Path(hdr_fname).stem}-{Path(sdr_fname).stem}.jpeg"
    img_write_8bit_jpeg_from_float(
        filename=gain_map_fname, img_float=gain_map_normalized
    )

    save_gain_map_metadata(
        hdr_fname=hdr_fname, sdr_fname=sdr_fname,
        offset_val=OFFSET_VAL,
        min_val=min_val, max_val=max_val,
        hdr_capacity_min=1.0,
        hdr_capacity_max=np.log2(1000/SDR_WHITE_LUMINANCE)
    )


def make_raw_for_ultrahdr_app(hdr_fname, sdr_fname, hdr_tf=tf.ST2084):
    png_16bit_to_rgba1010102(fname=hdr_fname)
    png_16bit_to_rgba8888(fname=sdr_fname)
    # _debug_calc_gain_map_metadata(hdr_fname=hdr_fname, sdr_fname=sdr_fname)
    make_sdr_8bit_jpeg(sdr_fname=sdr_fname)
    create_gain_map_jpeg_and_metadata(
        hdr_fname=hdr_fname, sdr_fname=sdr_fname, hdr_tf=hdr_tf
    )


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_test_data()
    # main_func(fname="./test_data.png")
    make_raw_for_ultrahdr_app(
        hdr_fname="./src_img/src_rec2100-pq.png",
        sdr_fname="./src_img/src_rec2020_srgb.png",
        hdr_tf=tf.ST2084
    )
    make_raw_for_ultrahdr_app(
        hdr_fname="./src_img/src_rec2100-hlg.png",
        sdr_fname="./src_img/src_rec2020_srgb.png",
        hdr_tf=tf.HLG
    )
