# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import imp
from pathlib import Path

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

SDR_CLIP_NAME = 'src_sdr_[0000-0023].png'
HDR_CLIP_NAME = 'src_hdr_[0000-0023].png'
EXR_CLIP_NAME = 'src_exr_[0000-0023].exr'


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


def explore_davinci_resolve_main():
    media_video_path = Path(
        'D:/abuse/2021/17_DaVinci_Resolve_CMS_Characteristics/src')
    # out_path = Path(
    #     'D:/Resolve/render_out/resolve_17_4')
    project_name = "Explore_DaVinci_CMS"
    # format_str = None
    # codec = None
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
    project_settings = {
        dcl.PRESET_KEY_COLOR_SCIENCE_MODE: dcl.RCM_YRGB_COLOR_MANAGED_V2,
        dcl.PRESET_KEY_RCM_PRESET_MODE: dcl.RCM_PRESET_SDR_709,
        dcl.PRESET_KEY_SEPARATE_CS_GM: dcl.RCM_SEPARATE_CS_GM_DISABLE
    }
    dcl.set_project_settings_from_dict(project, project_settings)

    # add items to media pool
    print("add media to pool")
    dcl.add_clips_to_media_pool(resolve, media_video_path)
    clip_obj_list, clip_name_list\
        = dcl.get_media_pool_clip_list_and_clip_name_list(project)
    print(f"clip_name_list = {clip_name_list}")

    # add video to the timeline
    clip_add_name_list = [
        SDR_CLIP_NAME, HDR_CLIP_NAME, EXR_CLIP_NAME]
    clip_add_obj_list = dcl.get_clip_obj_list_from_clip_name_list(
        clip_obj_list, clip_name_list, clip_add_name_list)
    dcl.create_timeline_from_clip(
        resolve, project, clip_add_obj_list, timeline_name="CMS")

    # set input color space for each clip
    dcl.set_clip_color_space(
        clip_obj_list, clip_name_list,
        SDR_CLIP_NAME, dcl.RCM_COLOR_SPACE_709_GM24)
    dcl.set_clip_color_space(
        clip_obj_list, clip_name_list,
        HDR_CLIP_NAME, dcl.RCM_COLOR_SPACE_2020_ST2084)
    dcl.set_clip_color_space(
        clip_obj_list, clip_name_list,
        EXR_CLIP_NAME, dcl.RCM_COLOR_SPACE_LINER)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
