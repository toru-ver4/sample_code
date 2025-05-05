# import standard libraries
import os

# import third-party libraries
import numpy as np
from imagecodecs import JPEGXR, imwrite, imread, jpegxr_encode

# my libraries
import transfer_functions as tf
import test_pattern_generator2 as tpg
import color_space as cs


def add_alpha_channel(image: np.ndarray) -> np.ndarray:
    """
    Add an alpha channel to an image if it has 3 channels (RGB).
    Parameters:
    image (np.ndarray): Input image array.
    Returns:
    np.ndarray: Image array with an added alpha channel if applicable.
    """
    if image.shape[-1] == 3:  # Check if the image has 3 channels (RGB)
        alpha_channel = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype)
        image = np.concatenate((image, alpha_channel), axis=-1)
    return image


def png_to_jxr(
        src_fname="./Windows_HDR_Capture/600.png",
        src_tf=tf.ST2084,
        src_cs=cs.BT2020
):
    if not JPEGXR.available:
        print("JPEG XR is not supported")
        return

    dst_fname = src_fname.replace(".png", ".jxr")

    image = tpg.img_read_as_float(src_fname)
    linear = tf.eotf(image, src_tf)
    large_xyz = cs.rgb_to_large_xyz(linear, src_cs)
    image = cs.large_xyz_to_rgb(large_xyz, cs.BT709)

    image *= 100 * 1.25  # 80 nits it 1.0
    image = add_alpha_channel(image)
    image = image.astype(np.float16)
    print(dst_fname, image.shape)
    imwrite(dst_fname, image, codec=jpegxr_encode)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    png_to_jxr(
        src_fname="./img/tp_st2084_bt2020.png",
        src_tf=tf.ST2084,
        src_cs=cs.BT2020
    )
