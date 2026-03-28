import os
import sys
from pathlib import Path

import numpy as np

import test_pattern_generator2 as tpg
import plot_utility as pu
import transfer_functions as tf

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from analyze_capture_data import (  # noqa: E402
    get_step_ramp_7colors,
    get_step_ramp_pos_list
)


def debug_ffmpeg_defference():
    ref_limited_img = tf.eotf_to_luminance(load_10bit_limited_ref_data(), tf.ST2084)
    hevc_img = tf.eotf_to_luminance(tpg.img_read_as_float("./debug/hevc_ffmpeg.png"), tf.ST2084)
    step_ramp_pos = get_step_ramp_pos_list(img_width=1920, img_height=1080)

    ref_step_ramp = get_step_ramp_7colors(ref_limited_img, step_ramp_pos)[0]
    hevc_step_ramp = get_step_ramp_7colors(hevc_img, step_ramp_pos)[0]

    ref_green = ref_step_ramp[:, 1].reshape(-1, 1)
    diff = hevc_step_ramp - ref_step_ramp
    diff_rate = diff / ref_green * 100

    fig, ax1 = pu.plot_1_graph(
        fontsize=16,
        figsize=(12, 6),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="(hevc_value - ref_value)/ref_value * 100",
        graph_title_size=None,
        xlabel="Luminance (nits)",
        ylabel="Error rate (%)",
        axis_label_size=None,
        legend_size=12,
        xlim=None,
        ylim=[-4, 4],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.set_xscale('log', base=10)
    color_list = [pu.RED, pu.GREEN, pu.BLUE]
    label_list = ["R", "G", "B"]
    for c_idx in range(3):
        ax1.plot(ref_green, diff_rate[:, c_idx].flatten(), "-o", color=color_list[c_idx], label=label_list[c_idx])
    pu.show_and_save(fig=fig, legend_loc='lower right', save_fname="./debug/step_ramp_diff.png", show=True)


def load_10bit_limited_ref_data():
    fname = "./debug/1920x1080_ST2084_Rec.2020.png"
    img = tpg.img_read_as_float(fname)
    img_limited_10bit = np.round((img * 219 + 16) * 4).astype(np.uint16)
    img_full = (img_limited_10bit - 64) / (940 - 64)

    return img_full


def create_playwrite_wrong_hdr_image():
    tpg.scrgb_jxr_to_rec2100_pq_png("./capture_img/right_profile_png_mdcv-p-None_mdcv-l-None_clli-None.jxr")
    tpg.scrgb_jxr_to_rec2100_pq_png("./capture_img/wrong_profile_png_mdcv-p-None_mdcv-l-None_clli-None.jxr")


def debug_avif_difference():
    tpg.scrgb_jxr_to_rec2100_pq_png("./debug/chrome_avif_mdcv-p-None_mdcv-l-None_clli-100.jxr")
    tpg.scrgb_jxr_to_rec2100_pq_png("./debug/edge_avif_mdcv-p-None_mdcv-l-None_clli-100.jxr")
    tpg.jxr_to_exr("./debug/chrome_avif_mdcv-p-None_mdcv-l-None_clli-100.jxr")
    tpg.jxr_to_exr("./debug/edge_avif_mdcv-p-None_mdcv-l-None_clli-100.jxr")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_ffmpeg_defference()
    debug_avif_difference()
    # create_playwrite_wrong_hdr_image()
