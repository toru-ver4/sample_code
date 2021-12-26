# -*- coding: utf-8 -*-
"""
import sys
sys.path.append("C:/Users/toruv/OneDrive/work/sample_code/2021/17_DaVinci_Resolve_CMS_Characteristics")
import davinci17_cms_explore as dce
import imp
imp.reload(dce)
# dce.explore_davinci_resolve_main()
dce.explore_davinci_resolve_main_ctrl()
"""

# import standard libraries
import os
import imp
from pathlib import Path
import re
# import logging
# import sys

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


# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename=os.path.join(os.getcwd(), "log.txt"), level=logging.INFO)
# sh = logging.StreamHandler(sys.stdout)
# logger.addHandler(sh)
# logger.setLevel(logging.WARNING)


def get_media_src_fname_sdr(idx=0):
    fname = MEDIA_SRC_PATH / f"src_sdr_{idx:04d}.png"
    return str(fname)


def get_media_src_fname_hdr(idx=0):
    fname = MEDIA_SRC_PATH / f"src_hdr_{idx:04d}.png"
    return str(fname)


def get_media_src_fname_exr(idx=0):
    fname = MEDIA_SRC_PATH / f"src_exr_{idx:04d}.exr"
    return str(fname)


def get_src_clip_name_prefix(src_name):
    return re.sub(r"_\[.*\]\..+$", "", src_name)


def make_output_path(
        src_name, out_dir, processing_mode, output_color_space):
    """
    Parameters
    ----------
    src_name : str
        src name
    out_dir : pathlib.Path
        output directory path
    processing_mode : str
        processing mode of DaVinci RCM.
    output_color_space : str
        output color space of DaVinci RCM.

    Examles
    -------
    >>> src_name = 'src_sdr_[0000-0001].png'
    >>> out_dir = Path('D:/abuse/dst')
    >>> processing_moede = 'HDR Rec.2020 Intermediate'
    >>> output_color_space = 'Rec.709 Gamma 2.4'
    """
    temp = src_name.replace("src", "dst")
    temp = re.sub(r"_\[.*\]\..+$", "", temp)
    base_name = temp + '--' + processing_mode + '--' + output_color_space
    base_name = base_name.replace(' ', "_")
    # print(f"base_name={base_name}")

    return out_dir / base_name


def make_project_settings(color_process_mode, output_color_space):
    if color_process_mode == dcl.RCM_PRESET_HDR_DAVINCI_INTERMEDIATE:
        timeline_working_lumiannce = dcl.TL_WORKING_LUMINANCE_4000
        timeline_working_luminance_mode = dcl.TL_WORKING_LUMINANCE_HDR_4000
        graphics_white_level = dcl.GRAPHICS_WHITE_LEVEL_200
        image_resize_gamma = dcl.IMAGE_RESIZE_GAMMA_LOG
    elif "SDR" in color_process_mode:
        timeline_working_lumiannce = dcl.TL_WORKING_LUMINANCE_100
        timeline_working_luminance_mode = dcl.TL_WORKING_LUMINANCE_SDR_100
        graphics_white_level = dcl.GRAPHICS_WHITE_LEVEL_100
        image_resize_gamma = dcl.IMAGE_RESIZE_GAMMA_GAMMA
    elif "HDR" in color_process_mode:
        timeline_working_lumiannce = dcl.TL_WORKING_LUMINANCE_1000
        timeline_working_luminance_mode = dcl.TL_WORKING_LUMINANCE_HDR_1000
        graphics_white_level = dcl.GRAPHICS_WHITE_LEVEL_200
        image_resize_gamma = dcl.IMAGE_RESIZE_GAMMA_LOG
    else:
        raise Exception("unknown color processing mode")

    timeline_color_space\
        = dcl.RCM_PRESEST_TO_TIMELINE_COLOR_SPACE[color_process_mode]
    output_gamma = dcl.OUTPUT_COLOR_SPACE_TO_GAMMA[output_color_space]
    project_sttings = {
        dcl.PRJ_SET_KEY_TIMELINE_FRAME_RATE: "24.0",
        dcl.PRJ_SET_KEY_TIMELINE_RESOLUTION_V: "1080",
        dcl.PRJ_SET_KEY_TIMELINE_RESOLUTION_H: "1920",
        dcl.PRJ_SET_KEY_VIDEO_MONITOR_FORMAT: "HD 1080p 24",
        # dcl.PRJ_SET_KEY_TIMELINE_PLAY_FRAME_RATE: "24",
        dcl.PRJ_SET_KEY_TIMELINE_WORKING_LUMINANCE: timeline_working_lumiannce,
        dcl.PRJ_SET_KEY_TIMELINE_WORKING_LUMINANCE_MODE:
            timeline_working_luminance_mode,
        dcl.PRJ_SET_KEY_TIMELINE_COLOR_SPACE: timeline_color_space,
        dcl.PRJ_SET_KEY_OUT_COLOR_SPACE: output_color_space,
        dcl.PRJ_SET_KEY_INPUT_DRT: dcl.INPUT_DRT_MODE_DAVINCI,
        dcl.PRJ_SET_KEY_OUTPUT_DRT: dcl.OUTPUT_DRT_MODE_DAVINCI,
        dcl.PRJ_SET_KEY_USE_INVERSE_DRT_FOR_SDR_TO_HDR: dcl.PRJ_VALUE_ENABLE,
        dcl.PRJ_SET_KEY_USE_WHITE_POINT_ADAPTATION: dcl.PRJ_VALUE_ENABLE,
        dcl.PRJ_SET_KEY_USE_CS_AWARE_GRADING_TOOLS: dcl.PRJ_VALUE_ENABLE,
        dcl.PRJ_SET_KEY_GRAPHICS_WHITE_LEVEL: graphics_white_level,
        dcl.PRJ_SET_KEY_IMAGE_RESIZE_GAMMA: image_resize_gamma,
        dcl.PRJ_SET_KEY_OUTPUT_GAMMA: output_gamma,
        dcl.PRJ_SET_KEY_COLOR_PROCESS_MODE: color_process_mode
    }

    return project_sttings


def explore_davinci_resolve_main(
        clip_name=HDR_CLIP_NAME,
        clip_color_space=dcl.RCM_COLOR_SPACE_2020_ST2084,
        color_process_mode=dcl.RCM_PRESET_HDR_2020_PQ_P3_LIMITED,
        output_color_space=dcl.RCM_COLOR_SPACE_709_GM24):
    media_video_path = MEDIA_SRC_PATH
    out_dir = MEDIA_DST_PATH
    export_dir = Path(
        'D:/abuse/2021/17_DaVinci_Resolve_CMS_Characteristics/export')
    project_name_prefix = get_src_clip_name_prefix(clip_name)
    project_name = project_name_prefix + '--' + color_process_mode\
        + '--' + output_color_space
    project_name = project_name.replace(" ", "_")
    format_str = dcl.OUT_FORMAT_TIF
    codec = dcl.CODEC_TIF_RGB16
    project_dir_name = "DaVinci_RCM_Research"

    # main process
    print("script start")
    resolve, project_manager = dcl.init_davinci17(
        close_current_project=True, delete_project_name=None,
        project_dir_name=project_dir_name)
    project = dcl.prepare_project(
        project_manager=project_manager,
        project_name=project_name)

    # project settings and preset
    project_settings_init = {
        dcl.PRJ_SET_KEY_COLOR_SCIENCE_MODE: dcl.RCM_YRGB_COLOR_MANAGED_V2,
        dcl.PRJ_SET_KEY_COLOR_PROCESS_MODE:
            dcl.RCM_PRESET_CUSTOM,
        dcl.PRJ_SET_KEY_SEPARATE_CS_GM: dcl.PRJ_VALUE_DISABLE
    }
    dcl.set_project_settings_from_dict(project, project_settings_init)

    project_sttings = make_project_settings(
        color_process_mode=color_process_mode,
        output_color_space=output_color_space)
    dcl.set_project_settings_from_dict(project, project_sttings)

    # restart project (save and load)
    # the reason this process is needed is below.
    # to verify the settings
    result = project_manager.SaveProject(project_name)
    if result is not True:
        raise Exception("save failed")
    print("script start")
    resolve, project_manager = dcl.init_davinci17(
        close_current_project=True, delete_project_name=None,
        project_dir_name=project_dir_name)
    project = dcl.prepare_project(
        project_manager=project_manager,
        project_name=project_name)

    # verify RCM preset name
    project_settings = project.GetSetting()
    rcm_preset = project_settings[dcl.PRJ_SET_KEY_COLOR_PROCESS_MODE]
    # dcl._debug_print_and_save_project_settings(
    #     project, log_file="./failed_settings.txt")
    if rcm_preset == color_process_mode:
        print("preset verify is OK")
    else:
        print("preset verity is NG")

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
        clip_obj_list, clip_name_list, clip_name, clip_color_space)

    # render settings
    out_path = make_output_path(
        clip_name, out_dir, color_process_mode, output_color_space)
    dcl.encode(resolve, project, out_path, format_str, codec)

    # save project
    result = project_manager.SaveProject()
    print(f"save result={result}")
    # export_file_path = str(export_dir/project_name)
    # print(f"export path = {export_file_path}")
    # result = project_manager.ExportProject(
    #     projectName=project_name,
    #     filePath=export_file_path,
    #     withStillsAndLUTs=False)
    # print(f"exporet result={result}")


def explore_davinci_resolve_main_ctrl():
    clip_name_list = [SDR_CLIP_NAME, HDR_CLIP_NAME, EXR_CLIP_NAME]
    clip_color_space_list = [
        dcl.RCM_COLOR_SPACE_709_GM24, dcl.RCM_COLOR_SPACE_2020_ST2084,
        dcl.RCM_COLOR_SPACE_LINER]
    color_process_mode_list = [
        dcl.RCM_PRESET_SDR_709,
        dcl.RCM_PRESET_SDR_2020,
        # dcl.RCM_PRESET_SDR_2020_P3_LIMITED,
        dcl.RCM_PRESET_SDR_P3_D60,
        # dcl.RCM_PRESET_HDR_DAVINCI_INTERMEDIATE,
        # dcl.RCM_PRESET_HDR_2020_INTERMEDIATE,
        dcl.RCM_PRESET_HDR_2020_HLG,
        # dcl.RCM_PRESET_HDR_2020_HLG_P3_LIMITED,
        dcl.RCM_PRESET_HDR_2020_PQ,
        # dcl.RCM_PRESET_HDR_2020_PQ_P3_LIMITED
    ]
    out_color_space_list = [
        dcl.RCM_COLOR_SPACE_709_GM24,
        dcl.RCM_COLOR_SPACE_2020_ST2084]

    for color_process_mode in color_process_mode_list:
        for out_color_space in out_color_space_list:
            for clip_name, clip_color_space\
                    in zip(clip_name_list, clip_color_space_list):
                print(f"{clip_color_space, color_process_mode}")
                explore_davinci_resolve_main(
                    clip_name=clip_name,
                    clip_color_space=clip_color_space,
                    color_process_mode=color_process_mode,
                    output_color_space=out_color_space)
                # break
            # break
        # break


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
