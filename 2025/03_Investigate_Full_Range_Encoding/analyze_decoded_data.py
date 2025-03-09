# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
from pathlib import Path

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# import my libraries
import test_pattern_generator2 as tpg
import plot_utility as pu
from create_src_test_pattern import calc_gradation_pattern_block_st_pos


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


def plot_full_data_with_diff(test_name, ramp_10bit_int, diff):
    # Create figure with two subplots arranged vertically (ax1 is top, ax2 is bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Define margin values for x and y axes
    x_margin = 10   # margin for x-axis
    y_margin_ax1 = 10  # margin for y-axis in ax1
    y_margin_ax2 = 0.5   # margin for y-axis in ax2

    # Set axis limits for ax1: x from 0 to 1023, y from 0 to 1023 (with margins)
    ax1.set_xlim(-x_margin, 1023 + x_margin)
    ax1.set_ylim(-y_margin_ax1, 1023 + y_margin_ax1)

    # Set axis limits for ax2: x from 0 to 1023, y from -10 to 10 (with margins)
    ax2.set_xlim(-x_margin, 1023 + x_margin)
    ax2.set_ylim(-4 - y_margin_ax2, 4 + y_margin_ax2)

    # Define custom tick positions for ax1
    xticks_ax1 = [x * 128 for x in range(8)] + [1023]
    yticks_ax1 = [x * 128 for x in range(8)] + [1023]
    ax1.set_xticks(xticks_ax1)
    ax1.set_yticks(yticks_ax1)
    ax1.set_xlabel("Code Value Before Encoding (10-bit)")
    ax1.set_ylabel("Code Value After Decoding (10-bit)")

    # Define custom tick positions for ax2
    xticks_ax2 = [x * 128 for x in range(8)] + [1023]
    yticks_ax2 = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    ax2.set_xticks(xticks_ax2)
    ax2.set_yticks(yticks_ax2)
    ax2.set_xlabel("Target Code Value (10-bit)")
    ax2.set_ylabel("Adjacent Difference")

    # Add grid lines (auxiliary lines) to both axes
    ax1.grid(True, which='both')
    ax2.grid(True, which='both')

    # Set titles with the appropriate English translations
    ax1.set_title(f"{test_name} Encode-Decode Result")
    ax2.set_title(f"{test_name} Adjacent Difference")

    # plot
    x = np.arange(1024)
    ax1.plot(x, ramp_10bit_int)
    ax2.plot(x[1:], diff)

    # Adjust layout and display the plot
    plt.tight_layout()
    # plt.show()
    save_fname = f"./img/{test_name}_with_diff.png"
    print(save_fname)
    plt.savefig(save_fname)


def plot_full_data_without_diff(test_name, ramp_10bit_int):

    # Create a figure with a single subplot (ax1)
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Define margin values for x and y axes
    x_margin = 10    # margin for x-axis
    y_margin = 10    # margin for y-axis

    # Set axis limits for ax1: x from 0 to 1023, y from 0 to 1023 (with margins)
    ax1.set_xlim(-x_margin, 1023 + x_margin)
    ax1.set_ylim(-y_margin, 1023 + y_margin)

    # Define custom tick positions for ax1
    xticks = [x * 128 for x in range(8)] + [1023]
    yticks = [x * 128 for x in range(8)] + [1023]
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)

    # Add grid lines (auxiliary lines) to ax1
    ax1.grid(True, which='both')

    # Set title with the appropriate English translation
    ax1.set_title(f"{test_name} Decoded 10-bit Ramp Result")

    x = np.arange(1024)
    ax1.plot(x, ramp_10bit_int)

    # Adjust layout and save the plot to a file
    plt.tight_layout()
    save_fname = f"./img/{test_name}.png"
    print(save_fname)
    plt.savefig(save_fname)


def check_decoded_full_range_data(test_name, decoded_png_fname, block_size):
    img = tpg.img_read_as_float(decoded_png_fname)

    pos_h, pos_v = calc_10bit_ramp_center_pos(width=width, block_size=block_size)

    ramp_10bit_float = img[pos_v, pos_h, 1]
    ramp_10bit_int = np.round(ramp_10bit_float * 1023).astype(np.int16)
    
    diff = ramp_10bit_int[1:] - ramp_10bit_int[:-1]

    plot_full_data_without_diff(
        test_name=test_name, ramp_10bit_int=ramp_10bit_float
    )
    plot_full_data_with_diff(
        test_name=test_name, ramp_10bit_int=ramp_10bit_int, diff=diff
    )


#####################
# Main
#####################
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    resolution = "1920x1080"
    width, _ = resolution.split("x")
    width = int(width)
    framerate = 24
    grey_block_size = 32

    encode_preset_list = [
        "./resolve_encode_preset/H265_MOV_Main10_Full.xml",
        "./resolve_encode_preset/H265_MOV_Main10_Limited.xml",
        "./resolve_encode_preset/H265_MP4_Main10_Full.xml",
        "./resolve_encode_preset/ProRes_MOV_422HQ_Full.xml",
        "./resolve_encode_preset/DNxHR_MOV_HQX_10-bit_Full.xml",
    ]

    for encode_preset in encode_preset_list:
        encode_preset_stem = Path(encode_preset).stem
        dir_path = Path("./encode_data/Resolve") / encode_preset_stem
        decoded_png = str(dir_path / encode_preset_stem) + "00086400.png"

        check_decoded_full_range_data(
            test_name=f"Resolve_{encode_preset_stem}",
            decoded_png_fname=decoded_png,
            block_size=grey_block_size
        )
