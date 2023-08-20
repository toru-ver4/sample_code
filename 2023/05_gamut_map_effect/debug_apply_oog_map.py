# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

# import third-party libraries

# import my libraries
from oog_map import debug_delta_rgb_p_using_hc_pattern_turbo
from jzazbz_azbz_czhz_plot import make_fname_cj_plane_with_interpolation
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def thread_wrappter_debug_apply_turbo_core(args):
    debug_apply_turbo_core(**args)


def debug_apply_turbo_core(h_idx):
    src_dir = "/work/overuse/2023/05_oog_color_map/oklab_cl_2020/"
    in_fname_base = src_dir + "ok_CL_w_lut_bt2020_bt709_{:04d}.png"
    dst_dir = "/work/overuse/2023/05_oog_color_map/oklab_cl_2020_colormap/"
    out_fname_base = dst_dir + "ok_CL_w_lut_bt2020_bt709_turbo_{:04d}.png"
    in_fname = in_fname_base.format(h_idx)
    out_fname = out_fname_base.format(h_idx)
    # print(in_fname)
    # print(out_fname)
    debug_delta_rgb_p_using_hc_pattern_turbo(
        in_fname=in_fname, out_fname=out_fname)


def thread_wrappter_debug_apply_turbo_jzazbz_core(args):
    debug_apply_turbo_jzazbz_core(**args)


def debug_apply_turbo_jzazbz_core(
        h_idx, dst_dir, color_space_name, maximum_luminance):
    in_fname = make_fname_cj_plane_with_interpolation(
        h_idx=h_idx, color_space_name=color_space_name,
        maximum_luminance=maximum_luminance)
    basename = Path(in_fname).name
    out_fname = dst_dir + basename
    print(in_fname)
    print(out_fname)
    debug_delta_rgb_p_using_hc_pattern_turbo(
        in_fname=in_fname, out_fname=out_fname)


def debug_apply_turbo_ctrl_to_oklab_cl_2020():
    src_dir = "/work/overuse/2023/05_oog_color_map/oklab_cl_2020"
    src_d = Path(src_dir)
    src_fname_list = src_d.glob("./*.png")

    total_process_num = len(list(src_fname_list))
    block_process_num = int(cpu_count() / 2)
    print(f"block_process_num {block_process_num}")
    block_num = int(round(total_process_num / block_process_num + 0.5))
    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            if h_idx >= total_process_num:                         # User
                break
            print(f"b_idx={b_idx}, p_idx={p_idx}, h_idx={h_idx}")  # User
            d = dict(
                h_idx=h_idx)
            # debug_apply_turbo_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(
                thread_wrappter_debug_apply_turbo_core, args)


def debug_apply_turbo_ctrl_to_jzazbz_cj_2020():
    src_dir = "/work/overuse/plot_seq/czjz_plane/"
    dst_dir = "/work/overuse/2023/05_oog_color_map/jzazbz_cj_2020_colormap/"
    color_space_name = cs.BT2020
    maximum_luminance = 100
    src_d = Path(src_dir)
    src_fname_list = src_d.glob("./*.png")

    total_process_num = len(list(src_fname_list))
    block_process_num = int(cpu_count() / 3)
    print(f"block_process_num {block_process_num}")
    block_num = int(round(total_process_num / block_process_num + 0.5))
    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            if h_idx >= total_process_num:                         # User
                break
            print(f"b_idx={b_idx}, p_idx={p_idx}, h_idx={h_idx}")  # User
            d = dict(
                h_idx=h_idx, dst_dir=dst_dir,
                color_space_name=color_space_name,
                maximum_luminance=maximum_luminance)
            # debug_apply_turbo_jzazbz_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(
                thread_wrappter_debug_apply_turbo_jzazbz_core, args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_apply_turbo_ctrl_to_oklab_cl_2020()
    debug_apply_turbo_ctrl_to_jzazbz_cj_2020()
