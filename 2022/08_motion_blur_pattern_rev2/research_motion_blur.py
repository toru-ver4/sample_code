# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def debug_save_blur_pattern():
    img = tpg.create_multi_border_tp(
        num_of_line=1, num_of_block=4,
        fg_color=[1, 1, 1], bg_color=[0, 0, 0],
        mag_rate=2)
    print(f"{img.shape[1]}x{img.shape[0]}")
    tpg.img_wirte_float_as_16bit_int(
        "./debug/multi_border.png", img)


def create_multi_border_tp_for_sight_move_or_integral(
        fg_color, bg_color, mag_rate=2):
    img = tpg.create_multi_border_tp(
        num_of_line=1, num_of_block=4,
        fg_color=fg_color, bg_color=bg_color, mag_rate=mag_rate)

    return img


def calc_line_st_pos_h_list(width, num_of_line):
    line_space_list = tpg.equal_devision(width, num_of_line)
    pos_h_list = [np.sum(line_space_list[:idx]) for idx in range(num_of_line)]

    return np.array(pos_h_list, dtype=np.uint16)


def draw_obj_with_routin_run(bg_img, fg_img, pos):
    _, bg_width = bg_img.shape[:2]
    fg_height, fg_width = fg_img.shape[:2]
    st_pos_h = pos[0]
    st_pos_v = pos[1]
    ed_pos_h = st_pos_h + fg_width
    ed_pos_v = st_pos_v + fg_height

    # for routin run
    ed_pos_h_2 = np.clip(ed_pos_h - bg_width, 0, bg_width)
    ed_pos_h = np.clip(ed_pos_h, 0, bg_width)

    fg_width_1 = ed_pos_h - st_pos_h
    fg_width_2 = ed_pos_h_2

    bg_img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h]\
        = fg_img[:, :fg_width_1]
    bg_img[st_pos_v:ed_pos_v, :ed_pos_h_2]\
        = fg_img[:, fg_width-fg_width_2:]


def thread_wrapper_create_sight_move_or_integral_pattern_seq_core(args):
    create_sight_move_or_integral_pattern_seq_core(**args)


def create_sight_move_or_integral_pattern_seq_core(
        f_idx, width, height, bg_color, fg_color, line_velocity_list,
        marker_velocity, moving_line_width, base_px_per_sec,
        num_of_line, out_dir, marker_mag_rate):
    # creeate moving pattern image, moving line image
    v_block_num = len(line_velocity_list)
    img = np.ones((height, width, 3)) * bg_color
    mp_img = create_multi_border_tp_for_sight_move_or_integral(
        fg_color=fg_color, bg_color=bg_color, mag_rate=marker_mag_rate)
    mp_img_velocity = np.array(marker_velocity)
    moving_line_height = height // v_block_num
    ml_img\
        = np.ones((moving_line_height, moving_line_width, 3)) * fg_color
    ml_img_velocity = np.array(line_velocity_list)
    line_pos_st_offset_h = (mp_img.shape[1] + ml_img.shape[1]) // 2
    line_pos_st_h = calc_line_st_pos_h_list(
        width=width, num_of_line=num_of_line)
    line_pos = np.zeros((v_block_num, num_of_line, 2))  # "2" is [pos_h, pos_v]
    for v_idx in range(v_block_num):
        line_pos[v_idx, :, 1] = v_idx * (height // v_block_num)
        line_pos[v_idx, :, 0]\
            = (line_pos_st_h + line_pos_st_offset_h
               + base_px_per_sec * ml_img_velocity[v_idx] * f_idx)\
            % width
    # line_pos[..., 0]\
    #     = (line_pos_st_h + line_pos_st_offset_h + base_px_per_sec * f_idx)\
    #     % width
    marker_pos = np.zeros((v_block_num, 2))  # "2" is [pos_h, pos_v]
    marker_pos[..., 1]\
        = np.arange(v_block_num) * (height // v_block_num)\
        + (moving_line_height // 2) - (mp_img.shape[0] // 2)
    marker_pos[:, 0] = (base_px_per_sec * mp_img_velocity * f_idx) % width
    # top marker position is center
    zero_idx = (mp_img_velocity <= 0.0)
    marker_pos[zero_idx, 0] = (width // 2) - (mp_img.shape[1] // 2)
    # print(line_pos)
    # print(marker_pos)

    for v_idx in range(v_block_num):
        for h_idx in range(num_of_line):
            draw_obj_with_routin_run(
                bg_img=img, fg_img=ml_img,
                pos=np.uint16(line_pos[v_idx, h_idx]))
        draw_obj_with_routin_run(
            bg_img=img, fg_img=mp_img, pos=np.uint16(marker_pos[v_idx]))

    pps_str = "-".join(
        [f"{int(x * base_px_per_sec):d}" for x in line_velocity_list])
    mvl_str = "-".join(
        [f"{int(x * base_px_per_sec):d}" for x in marker_velocity])

    fname = f"{out_dir}/sight_move_or_integral_{width}x{height}_"
    fname += f"nl-{num_of_line}_pps-{pps_str}_mvl-{mvl_str}_{f_idx:08d}.png"
    print(fname)
    img = np.uint8(np.round((img**(1/2.4)) * 0xFF))
    tpg.img_write(fname, img, 7)


def create_sight_move_or_integral_pattern_seq(
        width=3840, height=2160, base_px_per_sec=8, num_of_line=16,
        line_velocity_list=[1.0, 1.0, 1.5, 2.0, 2.5, 3.0],
        marker_velocity_list=[0, 1.0, 1.5, 2.0, 2.5, 3.0],
        marker_mag_rate=2, moving_line_width=2):
    num_of_frame = 600
    bg_color = np.array([0, 0, 0])
    fg_color = np.array([1, 1, 1])
    out_dir = "/work/overuse/2022/08_motion_blur_2/"
    out_dir += "sight_move_or_integral"

    total_process_num = num_of_frame
    block_process_num = int(cpu_count() * 0.8)
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            d = dict(
                f_idx=l_idx, width=width, height=height,
                bg_color=bg_color, fg_color=fg_color,
                moving_line_width=moving_line_width,
                line_velocity_list=line_velocity_list,
                marker_velocity=marker_velocity_list,
                base_px_per_sec=base_px_per_sec,
                num_of_line=num_of_line, out_dir=out_dir,
                marker_mag_rate=marker_mag_rate)
            args.append(d)
        #     create_sight_move_or_integral_pattern_seq_core(**d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(
                thread_wrapper_create_sight_move_or_integral_pattern_seq_core,
                args)


def debug_draw_obj_with_routin_run():
    bg_img = np.zeros((200, 200, 3))
    fg_img = create_multi_border_tp_for_sight_move_or_integral(
        fg_color=np.array([1, 1, 1]),
        bg_color=np.array([0, 0, 0]))
    draw_obj_with_routin_run(
        fg_img=fg_img, bg_img=bg_img, pos=[110, 25])
    tpg.img_wirte_float_as_16bit_int("./debug/routin_run.png", bg_img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_save_blur_pattern()
    # debug_draw_obj_with_routin_run()

    # create_sight_move_or_integral_pattern_seq(
    #     width=2160, height=3840, base_px_per_sec=8, num_of_line=4,
    #     v_block_num=2, marker_velocity_list=[0, 1.0])
    # create_sight_move_or_integral_pattern_seq(
    #     width=2160, height=3840, base_px_per_sec=16, num_of_line=4,
    #     line_velocity_list=[1.0, 1.5, 2.0, 2.5],
    #     marker_velocity_list=[0.0, 0.0, 0.0, 0.0])

    create_sight_move_or_integral_pattern_seq(
        width=3840, height=2160, base_px_per_sec=16, num_of_line=16,
        line_velocity_list=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        marker_velocity_list=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        marker_mag_rate=1)

    # create_sight_move_or_integral_pattern_seq(
    #     width=2160, height=3840, base_px_per_sec=16, num_of_line=4,
    #     line_velocity_list=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
    #     marker_velocity_list=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    #     marker_mag_rate=2, moving_line_width=4)
