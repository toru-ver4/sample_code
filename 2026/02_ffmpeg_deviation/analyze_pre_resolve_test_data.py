# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
from pathlib import Path

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from colour import write_image, read_image

# import my libraries
import test_pattern_generator2 as tpg
from create_src_test_pattern import get_10bit_ramp_from_img
from encode_decode_with_resolve import make_decode_output_fname
import plot_utility as pu


def check_davinci_resolve_encode_decode_data_core(fname):
    data = read_image(fname)
    rgb_10bit = np.round(get_10bit_ramp_from_img(img=data) * 1023).astype(np.int16)
    x = np.arange(1024, dtype=np.int16)
    ref_rgb10_bit = np.repeat(x[..., np.newaxis], 3, axis=-1)

    diff = rgb_10bit - ref_rgb10_bit
    channel_colors = [pu.RED, pu.GREEN, pu.BLUE]
    channel_labels = ["R", "G", "B"]
    x_ticks = [x * 64 for x in range(16)] + [1023]
    y_ticks_upper = [x * 64 for x in range(16)] + [1023]
    diff_y_ticks = list(range(-4, 5))

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [7, 3]}
    )
    ax1, ax2 = axes

    for c_idx, (color, label) in enumerate(zip(channel_colors, channel_labels)):
        ax1.plot(
            x, rgb_10bit[..., c_idx],
            '-', color=color, lw=1.5, label=f"Decoded {label}"
        )
        ax2.plot(
            x, diff[..., c_idx],
            '-o', color=color, lw=1.5, ms=4, label=label
        )

    for c_idx, (color, label) in enumerate(zip(channel_colors, channel_labels)):
        ax1.plot(
            x, ref_rgb10_bit[..., c_idx],
            '--', color=color, alpha=0.5, lw=2, label=f"Ref {label}"
        )


    title = Path(fname).name.replace("_00086400", "")
    ax1.set_title(title)
    ax1.set_ylabel("10-bit Code Value")
    ax1.set_xlim(0, 1023)
    ax1.set_ylim(-5, 1028)
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks_upper)
    ax1.grid(True, which='major', color="#B0B0B0", linestyle='-')
    ax1.legend(loc='upper left', ncol=2)

    ax2.axhline(0, color='k', lw=1, alpha=0.6)
    ax2.set_xlabel("Reference Code Value")
    ax2.set_ylabel("Decoded - Reference")
    ax2.set_xlim(0, 1023)
    ax2.set_xticks(x_ticks)
    ax2.set_ylim(-4.2, 4.2)
    ax2.set_yticks(diff_y_ticks)
    ax2.grid(True, which='major', color="#B0B0B0", linestyle='-')
    ax2.legend(loc='upper left', ncol=3)

    save_fname = f"./debug/end-resolve_dec-resolve_{title}"
    fig.tight_layout()
    plt.savefig(save_fname)
    plt.close(fig)


def check_davinci_resolve_encode_decode_data():
    encode_preset_list = [
        "./resolve_encode_preset/H.265_NVENC_Main10.xml",
        "./resolve_encode_preset/AV1_NVENC_Main10.xml",
    ]
    src_image_list = [
        "./img/src_img.dpx",
        "./img/src_img.png",
        "./img/src_img.tif"
    ]

    for encode_preset in encode_preset_list:
        for src_image in src_image_list:
            fname_base = make_decode_output_fname(
                src_image=src_image, encode_preset=encode_preset
            )
            fname = fname_base + "_00086400.png"
            print(fname)
            check_davinci_resolve_encode_decode_data_core(fname=fname)
        #     break
        # break
    

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    check_davinci_resolve_encode_decode_data()
