# -*- coding: utf-8 -*-

# import standard libraries
import subprocess
import sys
from pathlib import Path

# import third-party libraries
import numpy as np
from scipy import linalg

# import my libraries
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
TY_LIB_DIR = PROJECT_DIR.parents[1] / "ty_lib"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TY_LIB_DIR))

GRAY_PATCH_SIZE = 32

def vecmul(m, v):
    return np.matmul(m, v[..., None]).squeeze(-1)


def calc_block_num_h(width=1920, block_size=64):
    return width // block_size


def calc_gradation_pattern_block_st_pos(code_value, width, block_size):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_h = (code_value % block_num_h) * block_size
    st_pos_v = (code_value // block_num_h) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def calc_10bit_ramp_center_pos(width, block_size):
    pos_h_buf = []
    pos_v_buf = []
    
    for cv in range(1024):
        st_pos = calc_gradation_pattern_block_st_pos(
            code_value=cv, width=width, block_size=block_size
        )
        offset = block_size//2
        pos_h = st_pos[0] + offset
        pos_v = st_pos[1] + offset
        pos_h_buf.append(pos_h)
        pos_v_buf.append(pos_v)

    pos_h = np.array(pos_h_buf, dtype=np.uint16)
    pos_v = np.array(pos_v_buf, dtype=np.uint16)

    return pos_h, pos_v


def get_10bit_ramp_from_img(img: np.ndarray) -> np.ndarray:
    pos_h, pos_v = calc_10bit_ramp_center_pos(width=img.shape[1], block_size=GRAY_PATCH_SIZE)
    ramp_10bit = img[pos_v, pos_h]

    return ramp_10bit


def calc_rgb_to_ycbcr_matrix(gamut="bt.709"):
    if gamut == "bt.709":
        coef_y = np.array([0.2126, 0.7152, 0.0722])
    elif gamut == "bt.2020":
        coef_y = np.array([0.2627, 0.6780, 0.0593])
    else:
        raise ValueError("Invalid `gamut` parameter")
    div_cb = (coef_y[0] + coef_y[1]) * 2
    div_cr = (coef_y[1] + coef_y[2]) * 2
    coef_cb = (np.array([0.0, 0.0, 1.0]) - coef_y) / div_cb
    coef_cr = (np.array([1.0, 0.0, 0.0]) - coef_y) / div_cr

    mtx = np.vstack([coef_y, coef_cb, coef_cr])

    return mtx


def decode_yuv420p10le_1frame(in_fname, width=1920, height=1080):
    y_size = width * height
    uv_width = width // 2
    uv_height = height // 2
    uv_size = uv_width * uv_height
    total_samples = y_size + uv_size * 2

    frame = np.fromfile(in_fname, dtype='<u2', count=total_samples)
    if frame.size != total_samples:
        raise ValueError(
            f"Invalid frame size. expected={total_samples} samples, "
            f"actual={frame.size} samples."
        )

    y = frame[:y_size].reshape((height, width))
    u_420 = frame[y_size:y_size + uv_size].reshape((uv_height, uv_width))
    v_420 = frame[y_size + uv_size:].reshape((uv_height, uv_width))

    # 4:2:0 chroma planes are expanded back to luma resolution with NN.
    u = np.repeat(np.repeat(u_420, 2, axis=0), 2, axis=1)
    v = np.repeat(np.repeat(v_420, 2, axis=0), 2, axis=1)

    y = (y.astype(np.int16) - 64) / (219 * 4)
    u = (u.astype(np.int16) - 512) / (224 * 4)
    v = (v.astype(np.int16) - 512) / (224 * 4)

    mtx = linalg.inv(calc_rgb_to_ycbcr_matrix(gamut='bt.709'))
    yuv = np.dstack([y, u, v])
    rgb = vecmul(mtx, yuv)
    rgb_10bit = np.round(np.clip(rgb, 0.0, 1.0) * 1023).astype(np.uint16)

    return rgb_10bit


def extract_10bit_data_from_yuv420p10le(in_fname):
    rgb_10_bit = decode_yuv420p10le_1frame(in_fname=in_fname)
    ramp = get_10bit_ramp_from_img(img=rgb_10_bit)

    return ramp.astype(np.int16)


def run_command(args):
    print(" ".join(args))
    subprocess.run(args, check=True)


def create_yuv420p10le_using_ffmpeg():
    input_fname = "./img/src_img.png"
    output_fname = "./img/ffmpeg_yuv420p10le.yuv"
    cmd = "/opt/my_ffmpeg_out/bin/ffmpeg"
    ops = [
        "-hide_banner",
        "-i", input_fname,
        "-vf", "scale=in_range=full:out_range=limited:out_color_matrix=bt709",
        "-pix_fmt", "yuv420p10le",
        "-f", "rawvideo",
        output_fname,
        "-y"
    ]
    args = [cmd] + ops
    run_command(args)


if __name__ == '__main__':
    # ------------------------------------------------
    # create reference data and extract reference data
    # ------------------------------------------------
    # create_10bit_pattern_i010_format()
    ref_10_bit = extract_10bit_data_from_yuv420p10le("./raw/src_1920x1080_I010.yuv")

    # ------------------------------------------------
    # create target data using ffmpeg and extract target data
    # ------------------------------------------------
    # create_yuv420p10le_using_ffmpeg()
    target_10_bit = extract_10bit_data_from_yuv420p10le("./raw/ffmpeg_yuv420p10le.yuv")

    # ------------------------------------------------
    # check diff
    # ------------------------------------------------
    diff = np.abs(target_10_bit - ref_10_bit)
    max_diff = np.max(diff)
    if max_diff < 2:
        print(f"Maximum difference is {max_diff}")
        print("OK")
    else:
        print(f"Maximum difference is {max_diff}")
        print("NG")
