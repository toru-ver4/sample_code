# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import imp
from pathlib import Path
import re

# import third-party libraries

# import my libraries
import ty_davinci_control_lib as dcl
imp.reload(dcl)

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


TP_SRC_PATH = Path(
    'C:/Users/toruv/OneDrive/work/sample_code/2021/17_DaVinci_Resolve_CMS_Characteristics/img')
MEDIA_SRC_PATH = Path(
    'D:/abuse/2021/17_DaVinci_Resolve_CMS_Characteristics/src')
MEDIA_DST_PATH = Path(
    'D:/abuse/2021/17_DaVinci_Resolve_CMS_Characteristics/dst')
EXR_MIN_EXPOSURE = -6
EXR_MAX_EXPOSURE = 3

SDR_CLIP_NAME = 'src_sdr_[0000-0001].png'
HDR_CLIP_NAME = 'src_hdr_[0000-0001].png'
EXR_CLIP_NAME = 'src_exr_[0000-0001].exr'


"""
import sys
sys.path.append("C:/Users/toruv/OneDrive/work/sample_code/2021/17_DaVinci_Resolve_CMS_Characteristics")
import davinci17_cms_explore as dce
import imp
imp.reload(dce)
dce.explore_davinci_resolve_main()
"""

def get_media_src_fname_sdr(idx=0):
    fname = MEDIA_SRC_PATH / f"src_sdr_{idx:04d}.png"
    return str(fname)


def get_media_src_fname_hdr(idx=0):
    fname = MEDIA_SRC_PATH / f"src_hdr_{idx:04d}.png"
    return str(fname)


def get_media_src_fname_hdr(idx=0):
    fname = MEDIA_SRC_PATH / f"src_hdr_{idx:04d}.png"
    return str(fname)


def get_media_src_fname_exr(idx=0):
    fname = MEDIA_SRC_PATH / f"src_exr_{idx:04d}.exr"
    return str(fname)


def make_output_path(
        src_name, out_dir, processing_mode, output_color_space):
    temp = src_name.replace("src", "dst")
    temp = re.sub(r"_\[.*\]\..+$", "", temp)
    base_name = temp + '_' + processing_mode + '_' + output_color_space
    base_name = base_name.replace(' ', "_")
    print(f"base_name={base_name}")

    return out_dir / base_name


def explore_davinci_resolve_main():
    media_video_path = Path(
        'D:/abuse/2021/17_DaVinci_Resolve_CMS_Characteristics/src')
    out_dir = Path(
        'D:/abuse/2021/17_DaVinci_Resolve_CMS_Characteristics/dst')
    project_name = "Explore_DaVinci_CMS"
    clip_name = SDR_CLIP_NAME
    format_str = dcl.OUT_FORMAT_TIF
    codec = dcl.CODEC_TIF_RGB16
    color_process_mode = dcl.RCM_PRESET_HDR_DAVINCI_INTERMEDIATE
    output_color_space = dcl.RCM_COLOR_SPACE_2020_ST2084
    # # preset_name = H265_CQP0_PRESET_NAME

    # # main process
    # print("script start")
    resolve, project_manager = dcl.init_davinci17(
        close_current_project=True, delete_project_name=project_name)
    project = dcl.prepare_project(
        project_manager=project_manager,
        project_name=project_name)

    # project settings and preset
    # dcl._debug_print_and_save_project_settings(project)  # debug
    # dcl._debug_print_and_save_pretes(project)  # debug
    project_settings_init = {
        dcl.PRJ_SET_KEY_COLOR_SCIENCE_MODE: dcl.RCM_YRGB_COLOR_MANAGED_V2,
        dcl.PRJ_SET_KEY_COLOR_PROCESS_MODE:
            dcl.RCM_PRESET_HDR_DAVINCI_INTERMEDIATE,
        dcl.PRJ_SET_KEY_SEPARATE_CS_GM: dcl.RCM_SEPARATE_CS_GM_DISABLE
    }
    dcl.set_project_settings_from_dict(project, project_settings_init)

    project_sttings = {
        dcl.PRJ_SET_KEY_TIMELINE_FRAME_RATE: "24.0",
        dcl.PRJ_SET_KEY_TIMELINE_RESOLUTION_V: "1080",
        dcl.PRJ_SET_KEY_TIMELINE_RESOLUTION_H: "1920",
        dcl.PRJ_SET_KEY_VIDEO_MONITOR_FORMAT: "HD 1080p 24",
        dcl.PRJ_SET_KEY_TIMELINE_PLAY_FRAME_RATE: "24",
        dcl.PRJ_SET_KEY_COLOR_PROCESS_MODE: color_process_mode,
        dcl.PRJ_SET_KEY_OUT_COLOR_SPACE: output_color_space
    }
    dcl.set_project_settings_from_dict(project, project_sttings)

    # add items to media pool
    print("add media to pool")
    dcl.add_clips_to_media_pool(resolve, media_video_path)
    clip_obj_list, clip_name_list\
        = dcl.get_media_pool_clip_list_and_clip_name_list(project)
    print(f"clip_name_list = {clip_name_list}")

    # add video to the timeline
    clip_add_name_list = [clip_name]
    clip_add_obj_list = dcl.get_clip_obj_list_from_clip_name_list(
        clip_obj_list, clip_name_list, clip_add_name_list)
    dcl.create_timeline_from_clip(
        resolve, project, clip_add_obj_list, timeline_name="CMS")

    # set input color space for each clip
    dcl.set_clip_color_space(
        clip_obj_list, clip_name_list,
        clip_name, dcl.RCM_COLOR_SPACE_709_GM24)

    timeline = project.GetCurrentTimeline()
    timeline_settings = timeline.GetSetting()
    dcl._debug_save_dict_as_txt(
        "./timeline_settings.txt", timeline_settings)

    # # output settings
    # out_path = make_output_path(
    #     clip_name, out_dir, color_process_mode, output_color_space)
    # dcl.encode(resolve, project, out_path, format_str, codec)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
