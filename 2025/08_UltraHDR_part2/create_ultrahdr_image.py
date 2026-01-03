from pathlib import Path
import numpy as np
import os
import subprocess

from PIL import Image
import cv2

import test_pattern_generator2 as tpg
import transfer_functions as tf
import color_space as cs

GAIN_MAP_CS_NAME = cs.BT2020
# SDR_WHITE_LUMINANCE = 100
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


def linearize_input_image_for_8bit(fname, tf_name=tf.ST2084, cs_name=cs.P3_D65):
    img = tpg.img_read_as_float(fname)
    img = (np.round(img * 255))/255
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
    # img_pil.save(
    #     filename, 'JPEG', quality=100, subsampling=0
    # )


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
    rgba1010102.tofile(out_fname)

    return out_fname


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
    rgba8888.tofile(out_fname)

    return out_fname


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
        hdr_fname, sdr_fname, min_content_boost, max_content_boost, offset_val,
        hdr_capacity_min=0.0, hdr_capacity_max=2.3):
    cfg_name = f"./metadata/metadata_{Path(hdr_fname).stem}-"
    cfg_name += f"{Path(sdr_fname).stem}_hdr_capacity-{np.log2(hdr_capacity_max):.3f}.cfg"
    print(cfg_name)

    with open(cfg_name, 'wt') as f:
        buf = ""
        buf += f"--maxContentBoost {max_content_boost[0]} {max_content_boost[1]} {max_content_boost[2]}\n"
        buf += f"--minContentBoost {min_content_boost[0]} {min_content_boost[1]} {min_content_boost[2]}\n"
        buf += "--gamma 1.0 1.0 1.0\n"
        buf += f"--offsetSdr {offset_val} {offset_val} {offset_val}\n"
        buf += f"--offsetHdr {offset_val} {offset_val} {offset_val}\n"
        buf += f"--hdrCapacityMin {hdr_capacity_min}\n"
        buf += f"--hdrCapacityMax {hdr_capacity_max}\n"
        buf += "--useBaseColorSpace 1\n"
        f.write(buf)

    return cfg_name


def make_sdr_8bit_jpeg(sdr_fname: str):
    img = tpg.img_read_as_float(sdr_fname)
    fname = sdr_fname.replace(".png", "_8bit.jpeg")
    print(fname)
    img_write_8bit_jpeg_from_float(filename=fname, img_float=img)
    return fname


def create_gain_map_jpeg_and_metadata(hdr_fname, sdr_fname, hdr_tf, hdr_capacity_max=None):
    sdr_linear = linearize_input_image_for_8bit(
        fname=sdr_fname, tf_name=tf.SRGB, cs_name=cs.BT2020
    )
    sdr_linear = sdr_linear * (SDR_WHITE_LUMINANCE / tf.REF_WHITE_LUMINANCE)

    hdr_linear = linearize_input_image(
        fname=hdr_fname, tf_name=hdr_tf, cs_name=cs.BT2020
    )

    gain_map_raw = np.log2((hdr_linear + OFFSET_VAL)/(sdr_linear + OFFSET_VAL))
    # gain_map_raw[gain_map_raw < 0] = 0.0

    gain_map_normalized = np.zeros_like(gain_map_raw)
    min_val = np.min(gain_map_raw, axis=(0, 1))
    max_val = np.max(gain_map_raw, axis=(0, 1))
    print(min_val, max_val)

    for c_idx in range(3):
        gain_map_normalized[..., c_idx] =\
            np.clip((gain_map_raw[..., c_idx] - min_val[c_idx]) / (max_val[c_idx] - min_val[c_idx]), 0.0, 1.0)

    gain_map_fname = "./gain_map_img/gain_map_"
    gain_map_fname += f"{Path(hdr_fname).stem}-{Path(sdr_fname).stem}.jpeg"
    img_write_8bit_jpeg_from_float(
        filename=gain_map_fname, img_float=gain_map_normalized
    )

    if hdr_capacity_max is None:
        hdr_capacity_max = np.log2(np.max(hdr_linear)/(SDR_WHITE_LUMINANCE / tf.REF_WHITE_LUMINANCE))

    metadata_fname = \
        save_gain_map_metadata(
            hdr_fname=hdr_fname, sdr_fname=sdr_fname,
            offset_val=OFFSET_VAL,
            min_content_boost=2**min_val,
            max_content_boost=2**max_val,
            hdr_capacity_min=2**0.0,
            hdr_capacity_max=2**hdr_capacity_max
        )

    return gain_map_fname, metadata_fname


def craete_files_for_ultrahdr_app_scenario4(hdr_fname, sdr_fname, hdr_tf=tf.ST2084, hdr_capacity_max=None):
    sdr_jpeg_fname = make_sdr_8bit_jpeg(sdr_fname=sdr_fname)
    gain_map_fname, metadata_fname = \
        create_gain_map_jpeg_and_metadata(
            hdr_fname=hdr_fname, sdr_fname=sdr_fname, hdr_tf=hdr_tf,
            hdr_capacity_max=hdr_capacity_max
        )

    output_fname = f"./gain_map_img/{Path(sdr_fname).stem}-{Path(metadata_fname).stem}.jpg"

    run_ultrahdr_app_scenario_4(
        sdr_fname=sdr_jpeg_fname, gain_map_fname=gain_map_fname,
        output_fname=output_fname, metadata_fname=metadata_fname
    )


def craete_files_for_ultrahdr_app_scenario1(hdr_fname, sdr_fname):
    img = tpg.img_read(hdr_fname)
    img_height, img_width = img.shape[:2]

    hdr_raw_fname = png_16bit_to_rgba1010102(fname=hdr_fname)
    sdr_raw_fname = png_16bit_to_rgba8888(fname=sdr_fname)

    output_fname = f"./gain_map_img/{Path(sdr_fname).stem}-{Path(hdr_fname).stem}_scenario1.jpg"

    run_ultrahdr_app_scenario_1(
        hdr_raw_fname=hdr_raw_fname, sdr_raw_fname=sdr_raw_fname, output_fname=output_fname,
        img_width=img_width, img_height=img_height
    )


def run_ultrahdr_app_scenario_1(
        hdr_raw_fname, sdr_raw_fname, output_fname, img_width, img_height):
    cmd = [
        "ultrahdr_app",
        "-m", "0",  # 0: encode, 1: decode
        "-p", hdr_raw_fname,  # raw hdr source
        "-a", "5",  # 0:p010, 4: rgbahalffloat, 5:rgba1010102
        "-y", sdr_raw_fname,  # raw sdr source
        "-b", "3",  # 1:yuv420, 3:rgba8888 (default)
        "-w", str(img_width),  # input file width
        "-h", str(img_height),  # input file height
        "-q", "100",  # quality parameter for sdr
        "-Q", "100",  # quality parameter for gain map
        "-C", "2",  # hdr intent color gamut. 0: bt709, 1: p3, 2: bt2100
        "-c", "2",  # sdr intent color gamut. 0: bt709, 1: p3, 2: bt2100
        "-t", "2",  # hdr_intent_transfer function. 0:linear, 1: hlg, 2:pq
        "-R", "1",  # color range. 0: narrow range, 1: full range
        "-M", "1",  # multi channel gain map. 0: disable, 1: enable
        # "-L", "1000",  # target display peak luminance [nits].
        "-z", output_fname  # output file
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("ultrahdr_app output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running ultrahdr_app:", e.stderr)


def run_ultrahdr_app_scenario_4(
        sdr_fname, gain_map_fname, output_fname, metadata_fname):
    cmd = [
        "ultrahdr_app",
        "-m", "0",  # 0: encode, 1: decode
        "-i", sdr_fname,  # sdr source
        "-g", gain_map_fname,  # gain map
        "-q", "100",  # quality parameter for sdr
        "-Q", "100",  # quality parameter for gain map
        "-D", "1",  # encoding preset. 0:real time, 1:best quality
        "-C", "2",  # hdr intent color gamut. 0: bt709, 1: p3, 2: bt2100
        "-c", "2",  # sdr intent color gamut. 0: bt709, 1: p3, 2: bt2100
        "-t", "2",  # hdr_intent_transfer function. 0:linear, 1: hlg, 2:pq
        "-R", "1",  # color range. 0: narrow range, 1: full range
        "-M", "1",  # multi channel gain map. 0: disable, 1: enable
        # "-L", "1000",  # target display peak luminance [nits].
        "-f", metadata_fname,  # gainmap metadata
        "-z", output_fname  # output file
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("ultrahdr_app output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running ultrahdr_app:", e.stderr)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_test_data()
    # convert_avif_to_png_2100pq_2020srgb_river()
    # convert_avif_to_png_rec2100_pq_shiga_kougen()

    # craete_files_for_ultrahdr_app(
    #     hdr_fname="./src_png/1920x1080_ST2084_Rec.2020.png",
    #     sdr_fname="./src_png/1920x1080_sRGB_Rec.2020_0.5x.png",
    #     hdr_tf=tf.ST2084
    # )

    # craete_files_for_ultrahdr_app_scenario4(
    #     hdr_fname="./src_png/1920x1080_ST2084_Rec.2020.png",
    #     sdr_fname="./src_png/1920x1080_sRGB_Rec.2020_0.5x.png",
    #     hdr_tf=tf.ST2084,
    #     hdr_capacity_max=1.0
    # )

    # craete_files_for_ultrahdr_app_scenario4(
    #     hdr_fname="./src_png/1920x1080_ST2084_Rec.2020.png",
    #     sdr_fname="./src_png/1920x1080_sRGB_Rec.2020_0.5x.png",
    #     hdr_tf=tf.ST2084,
    #     hdr_capacity_max=5.622376462364273
    # )

    # craete_files_for_ultrahdr_app_scenario4(
    #     hdr_fname="./src_png/HDR_Capacity_2.300_1280x720.png",
    #     sdr_fname="./src_png/HDR_Capacity_SDR_1280x720.png",
    #     hdr_tf=tf.ST2084,
    #     hdr_capacity_max=2.3
    # )

    # craete_files_for_ultrahdr_app_scenario4(
    #     hdr_fname="./src_png/1920x1080_ST2084_Rec.2020.png",
    #     sdr_fname="./src_png/lut_check_pattern_sdr.png",
    #     hdr_tf=tf.ST2084,
    #     hdr_capacity_max=1.0
    # )

    # craete_files_for_ultrahdr_app_scenario4(
    #     hdr_fname="./src_png/lut_check_pattern_hdr.png",
    #     sdr_fname="./src_png/lut_check_pattern_sdr.png",
    #     hdr_tf=tf.ST2084,
    #     hdr_capacity_max=1.0
    # )

    # craete_files_for_ultrahdr_app_scenario1(
    #     hdr_fname="./src_png/1920x1080_ST2084_Rec.2020.png",
    #     sdr_fname="./src_png/1920x1080_sRGB_Rec.2020_0.5x.png"
    # )

    # craete_files_for_ultrahdr_app_scenario1(
    #     hdr_fname="./src_png/1920x1080_ST2084_Rec.2020.png",
    #     sdr_fname="./src_png/1920x1080_sRGB_Rec.2020.png"
    # )

    # hdr_capacity_list = [
    #     0.563, 0.978, 1.300, 1.563, 1.978, 2.300, 2.563, 2.885, 3.300, 3.622, 3.885, 4.300, 5.622, 6.965
    # ]
    # for hdr_capacity in hdr_capacity_list:
    #     print(hdr_capacity)

    #     craete_files_for_ultrahdr_app_scenario4(
    #         hdr_fname=f"./src_png/HDR_Capacity_{hdr_capacity:.3f}_1280x720.png",
    #         sdr_fname="./src_png/HDR_Capacity_SDR_1280x720.png",
    #         hdr_tf=tf.ST2084,
    #         hdr_capacity_max=hdr_capacity
    #     )

    tpg.png_to_avif_2(
        png_fname="./src_png/1920x1080_ST2084_Rec.2020.png",
        avif_fname="./src_png/1920x1080_ST2084_Rec.2020_ISO_HDR.avif",
        color_space_name=cs.BT2020,
        transfer_characteristics=tf.ST2084,
        lossless=True,
        cll=0, pall=0
    )
