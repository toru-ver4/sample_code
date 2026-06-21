# -*- coding: utf-8 -*-

# import standard libraries
import sys
from pathlib import Path

# import third-party libraries
import numpy as np
from colour import read_image

# import my libraries
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
TY_LIB_DIR = PROJECT_DIR.parents[1] / "ty_lib"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TY_LIB_DIR))

from create_src_test_pattern2 import (
    COLOR_LIST,
    get_10bit_ramp_from_img,
)


BIT_DEPTH_LIST = [8, 10, 12]
GAMUT_LIST = ["bt.709", "bt.2020"]
SUBSAMPLING_LIST = ["420", "422", "444"]
GRAY_TOLERANCE = 1
COLOR_TOLERANCE = 2


def make_condition_list():
    condition_list = []
    for bit_depth in BIT_DEPTH_LIST:
        for subsampling in SUBSAMPLING_LIST:
            for gamut in GAMUT_LIST:
                condition_list.append(
                    {
                        "bit_depth": bit_depth,
                        "subsampling": subsampling,
                        "gamut": gamut,
                        "file_pix_fmt": make_file_pix_fmt(
                            subsampling=subsampling, bit_depth=bit_depth
                        ),
                    }
                )

    return condition_list


def make_file_pix_fmt(subsampling, bit_depth):
    return f"yuv{subsampling}p{bit_depth}le"


def float_to_n_bit(xx, max_cv):
    return np.rint(xx * max_cv).astype(np.int32)


def mask_color(xx, color_list):
    for ii, color in enumerate(color_list):
        for jj in range(len(color)):
            xx[ii, :, jj] = xx[ii, :, jj] * color[jj]


def create_ref_ramp_data(bit_depth):
    num_of_color = len(COLOR_LIST)
    num_of_patch = 2 ** bit_depth

    x = np.arange(num_of_patch, dtype=np.int32)
    ref_data = np.repeat(x[..., np.newaxis], 3, axis=-1)
    ref_data = np.repeat(ref_data[np.newaxis, ...], num_of_color, axis=0)
    mask_color(ref_data, COLOR_LIST)

    return ref_data


def make_encode_eval_rgb_fname(file_pix_fmt, gamut):
    return f"./img/de265_decode_{file_pix_fmt}_{gamut}.dpx"


def make_decode_eval_rgb_fname(file_pix_fmt, gamut):
    return f"./img/ffmpeg_decode_x265_{file_pix_fmt}_{gamut}.dpx"


def read_ramp_from_image(img_fname, bit_depth):
    if not Path(img_fname).exists():
        raise FileNotFoundError(img_fname)

    max_cv = (2 ** bit_depth) - 1
    read_data_float = get_10bit_ramp_from_img(read_image(img_fname), bit_depth)

    return float_to_n_bit(read_data_float, max_cv)


def calc_ramp_diff(read_data, ref_data):
    diff = np.abs(read_data.astype(np.int32) - ref_data.astype(np.int32))
    gray_max_diff = int(np.max(diff[0]))
    color_max_diff = int(np.max(diff[1:]))

    return gray_max_diff, color_max_diff


def check_condition(gray_diff, color_diff):
    return gray_diff <= GRAY_TOLERANCE and color_diff <= COLOR_TOLERANCE


def print_result(direction, bit_depth, file_pix_fmt, gamut, gray_diff, color_diff):
    result = "OK" if check_condition(gray_diff, color_diff) else "NG"
    print(
        f"[{result}] {direction:6} {bit_depth:2d}-bit "
        f"{file_pix_fmt:11} {gamut:7} "
        f"gray_diff = {gray_diff}, color_diff = {color_diff}"
    )


def evaluate_encode(condition, ref_data):
    bit_depth = condition["bit_depth"]
    gamut = condition["gamut"]
    file_pix_fmt = condition["file_pix_fmt"]
    rgb_fname = make_encode_eval_rgb_fname(file_pix_fmt=file_pix_fmt, gamut=gamut)

    read_data = read_ramp_from_image(img_fname=rgb_fname, bit_depth=bit_depth)

    return calc_ramp_diff(read_data=read_data, ref_data=ref_data)


def evaluate_decode(condition, ref_data):
    bit_depth = condition["bit_depth"]
    gamut = condition["gamut"]
    file_pix_fmt = condition["file_pix_fmt"]
    rgb_fname = make_decode_eval_rgb_fname(file_pix_fmt=file_pix_fmt, gamut=gamut)

    read_data = read_ramp_from_image(img_fname=rgb_fname, bit_depth=bit_depth)

    return calc_ramp_diff(read_data=read_data, ref_data=ref_data)


def main():
    all_ok = True

    for condition in make_condition_list():
        bit_depth = condition["bit_depth"]
        gamut = condition["gamut"]
        file_pix_fmt = condition["file_pix_fmt"]
        ref_data = create_ref_ramp_data(bit_depth=bit_depth)

        gray_diff, color_diff = evaluate_encode(condition=condition, ref_data=ref_data)
        print_result(
            direction="encode", bit_depth=bit_depth, file_pix_fmt=file_pix_fmt,
            gamut=gamut, gray_diff=gray_diff, color_diff=color_diff
        )
        all_ok = all_ok and check_condition(gray_diff, color_diff)

        gray_diff, color_diff = evaluate_decode(condition=condition, ref_data=ref_data)
        print_result(
            direction="decode", bit_depth=bit_depth, file_pix_fmt=file_pix_fmt,
            gamut=gamut, gray_diff=gray_diff, color_diff=color_diff
        )
        all_ok = all_ok and check_condition(gray_diff, color_diff)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
