# import standard libraries
import os

# import third-party libraries
import numpy as np
from imagecodecs import JPEGXR, imread
from colour.io import write_image

# my libraries
import transfer_functions as tf
import test_pattern_generator2 as tpg
import color_space as cs


def jxr_to_exr(src_fname="./Windows_HDR_Capture/600.jxr"):
    if not JPEGXR.available:
        print("JPEG XR is not supported")
        return

    dst_fname = src_fname.replace(".jxr", ".exr")
    image = imread(src_fname) * 0.8
    image = image[..., :3]  # remove alpha
    print(image.dtype)
    write_image(image=image, path=dst_fname)


def jxr_to_avif_lossy(src_fname="./Windows_HDR_Capture/600.jxr"):
    if not JPEGXR.available:
        print("JPEG XR is not supported")
        return

    avif_fname = src_fname.replace(".jxr", ".avif")
    png_fname = src_fname.replace(".jxr", ".png")
    image = imread(src_fname) * 80
    image = image[..., :3]  # remove alpha
    image_xyz = cs.rgb_to_large_xyz(image, cs.BT709)
    image_rgb = cs.large_xyz_to_rgb(image_xyz, cs.BT2020)
    image_rgb = np.clip(image_rgb, 0, 10000)
    out_img = tf.oetf_from_luminance(image_rgb, tf.ST2084)
    print(png_fname)
    tpg.img_wirte_float_as_16bit_int(png_fname, out_img, comp_val=6)
    tpg.png_to_avif_2(
        png_fname=png_fname, avif_fname=avif_fname,
        color_space_name=cs.BT2020, transfer_characteristics=tf.ST2084,
        lossless=False, cll=10000, pall=10000
    )


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # jxr_to_exr("./capture_data/check_precision/Microsoft_Edge.jxr")
    # jxr_to_avif_lossy("./capture_data/check_precision/Microsoft_Edge.jxr")
    # jxr_to_exr("./capture_data/check_precision/Monster_Hunter_Wilds.jxr")
    # jxr_to_avif_lossy("./capture_data/check_precision/Monster_Hunter_Wilds.jxr")

    jxr_to_exr("./capture_data/01_obs_internal_tonemapping/screenshot.jxr")
