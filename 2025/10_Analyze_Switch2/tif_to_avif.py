from pathlib import Path
import os
import glob
import sys

import numpy as np
import cv2

import test_pattern_generator2 as tpg
import transfer_functions as tf
import color_space as cs


def tif_to_png(tif_fname="./debug/blog/tif/IMG-05_P1-23_P2-00.tif"):
    png_fname_base = Path(tif_fname).stem
    png_fname = str(Path("./debug/blog/png") / png_fname_base) + ".png"
    img = tpg.img_read(tif_fname)
    img_resized = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    print(f"{tif_fname} -> {png_fname}")
    tpg.img_write(png_fname, img_resized, comp_val=6)

    info_txt = "./luminance.txt"
    with open(info_txt, "a") as f:
        pass
        linear = tf.eotf_to_luminance(img_resized/65535, tf.ST2084)
        large_y = cs.rgb_to_large_xyz(linear, color_space_name=cs.BT2020)[..., 1]
        max_large_y = np.max(large_y)
        ave_y = np.mean(large_y)
        f.write(f"{png_fname}: max = {max_large_y:.1f} nits, ave = {ave_y:.1f}\n")
        
    return png_fname


def png_to_avif(png_fname):
    avif_fname_base = Path(png_fname).stem
    avif_fname = str(Path("./debug/blog/avif") / avif_fname_base) + ".avif"
    print(avif_fname)

    tpg.png_to_avif_2(
        png_fname=png_fname,
        avif_fname=avif_fname,
        lossless=False,
        cll=0, pall=0
    )


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # for tif_fname in glob.glob("./debug/blog/tif/*.tif"):
    #     png_fname = tif_to_png(tif_fname)
    #     png_to_avif(png_fname)
    # png_to_avif(png_fname="./debug/blog/png/MKW-02_P1-000_P2-10.png")
    # png_to_avif(png_fname="./debug/blog/png/MKW-02_P1-992_P2-00.png")
    png_to_avif(png_fname="./debug/blog/png/MKW-01_P1-000_P2-10.png")
    png_to_avif(png_fname="./debug/blog/png/MKW-01_P1-092_P2-00.png")
