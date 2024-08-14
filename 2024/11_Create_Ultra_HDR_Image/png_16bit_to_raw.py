import cv2
import numpy as np
import os

import test_pattern_generator2 as tpg


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
    # print(np.vectorize(hex)(rgba1010102))

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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_test_data()
    # main_func(fname="./test_data.png")
    png_16bit_to_rgba1010102(fname="./src_rec2100-pq.png")
    png_16bit_to_rgba8888(fname="./src_rec709.png")
