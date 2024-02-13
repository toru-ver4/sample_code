# -*- coding: utf-8 -*-
"""
A library for DaVinci Resolve control
=====================================

"""

# import standard libraries
import os
import time
import pprint
from pathlib import Path
import re

# import third-party libraries
from python_get_resolve import GetResolve

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

resolve = GetResolve()


"""
Example 1
---------
import imp
import ty_davinci_control_lib as dv_lib
imp.reload(dv_lib)
dv_lib.test_func(close_current_project=True)

Example 2
---------
import imp
import ty_davinci_control_lib as dv_lib
imp.reload(dv_lib)
project = dv_lib.test_func()
...
dv_lib._debug_print_and_save_project_settings(project)

Example 3
import sys
sys.path.append("C:/Users/toruv/OneDrive/work/sample_code/ty_lib")
import imp
import ty_davinci_control_lib as dv_lib
imp.reload(dv_lib)
dv_lib.get_avilable_parameters()
"""

# Page
MEDIA_PAGE_STR = "media"
CUT_PAGE_STR = "cut"
EDIT_PAGE_STR = "edit"
FUSION_PAGE_STR = 'fusion'
COLOR_PAGE_STR = "color"
DELIVER_PAGE_STR = "deliver"

# Project Settings
PRJ_PARAM_NONE = "None"
PRJ_PARAM_ENABLE = "1"
PRJ_PARAM_DISABLE = "1"

PRJ_TIMELINE_RESOLUTION_1920 = "1920"
PRJ_TIMELINE_RESOLUTION_1080 = "1080"
PRJ_TIMELINE_RESOLUTION_3840 = "3840"
PRJ_TIMELINE_RESOLUTION_2160 = "2160"

PRJ_VIDEO_MONITOR_FORMAT_HD_1080P23FPS = "HD 1080p 23.976"
PRJ_VIDEO_MONITOR_FORMAT_HD_1080P24FPS = "HD 1080p 24"
PRJ_VIDEO_MONITOR_FORMAT_HD_1080P29FPS = "HD 1080p 29.97"
PRJ_VIDEO_MONITOR_FORMAT_HD_1080P25FPS = "HD 1080p 25"
PRJ_VIDEO_MONITOR_FORMAT_HD_1080P30FPS = "HD 1080p 30"
PRJ_VIDEO_MONITOR_FORMAT_HD_1080P50FPS = "HD 1080p 50"
PRJ_VIDEO_MONITOR_FORMAT_HD_1080P59FPS = "HD 1080p 59.94"
PRJ_VIDEO_MONITOR_FORMAT_HD_1080P60FPS = "HD 1080p 60"

PRJ_VIDEO_MONITOR_FORMAT_UHD_2160P23FPS = "UHD 2160p 23.976"
PRJ_VIDEO_MONITOR_FORMAT_UHD_2160P24FPS = "UHD 2160p 24"
PRJ_VIDEO_MONITOR_FORMAT_UHD_2160P25FPS = "UHD 2160p 25"
PRJ_VIDEO_MONITOR_FORMAT_UHD_2160P29FPS = "UHD 2160p 29.97"
PRJ_VIDEO_MONITOR_FORMAT_UHD_2160P30FPS = "UHD 2160p 30"
PRJ_VIDEO_MONITOR_FORMAT_UHD_2160P50FPS = "UHD 2160p 50"
PRJ_VIDEO_MONITOR_FORMAT_UHD_2160P59FPS = "UHD 2160p 59.94"
PRJ_VIDEO_MONITOR_FORMAT_UHD_2160P60FPS = "UHD 2160p 60"

PRJ_TIMELINE_FRAMERATE_23 = "23.976"
PRJ_TIMELINE_FRAMERATE_24 = "24.0"
PRJ_TIMELINE_FRAMERATE_25 = "25.0"
PRJ_TIMELINE_FRAMERATE_29 = "29.97"
PRJ_TIMELINE_FRAMERATE_30 = "30.0"
PRJ_TIMELINE_FRAMERATE_50 = "50.0"
PRJ_TIMELINE_FRAMERATE_59 = "59.94"
PRJ_TIMELINE_FRAMERATE_60 = "60.0"

PRJ_TIMELINE_PLAYBACK_FRAMERATE_23 = "23.976"
PRJ_TIMELINE_PLAYBACK_FRAMERATE_24 = "24"
PRJ_TIMELINE_PLAYBACK_FRAMERATE_25 = "25"
PRJ_TIMELINE_PLAYBACK_FRAMERATE_29 = "29.97"
PRJ_TIMELINE_PLAYBACK_FRAMERATE_30 = "30"
PRJ_TIMELINE_PLAYBACK_FRAMERATE_50 = "50"
PRJ_TIMELINE_PLAYBACK_FRAMERATE_59 = "59.94"
PRJ_TIMELINE_PLAYBACK_FRAMERATE_60 = "60"

PRJ_VIDEO_DATA_LEVEL_LIMITED = "Video"
PRJ_VIDEO_DATA_LEVEL_FULL = "Full"

PRJ_COLOR_SCIENCE_MODE_RCM_OFF = "davinciYRGB"
PRJ_COLOR_SCIENCE_MODE_RCM_ON = "davinciYRGBColorManagedv2"
PRJ_COLOR_SCIENCE_MODE_ACES_CC = "acescc"
PRJ_COLOR_SCIENCE_MODE_ACES_CCT = "acescct"

PRJ_PRESET_MODE_CUSTOM = "Custom"

PRJ_GAMMA_STR_ST2084 = "ST2084"
PRJ_GAMMA_STR_GAMMA24 = "Gamma 2.4"

PRJ_COLOR_SPACE_REC2020 = "Rec.2020"
PRJ_COLOR_SPACE_REC709 = "Rec.709"

PRJ_ACES_ODT_P3D65_PQ_108 = "P3-D65 ST2084 (108 nits)"
PRJ_ACES_ODT_P3D65_PQ_1000 = "P3-D65 ST2084 (1000 nits)"
PRJ_ACES_ODT_P3D65_PQ_4000 = "P3-D65 ST2084 (4000 nits)"

PRJ_ACES_ODT_REC2020_PQ_1000 = "Rec.2020 ST2084 (1000 nits)"


def wait_for_rendering_completion(project):
    while project.IsRenderingInProgress():
        time.sleep(1)
    return


def get_resolve():
    return resolve


def close_and_remove_project(project_name):
    project_manager = resolve.GetProjectManager()
    project_manager.CloseProject(project_manager.GetCurrentProject())
    project_manager.DeleteProject(project_name)


def init_resolve(close_current_project=True):
    """
    Initialize davinci17 python environment.

    * create ProjectManager instance
    * close current project for initialize if needed.

    Returns
    -------
    resolve
        Resolve instance
    project_namager
        ProjectManager instance
    """
    project_manager = resolve.GetProjectManager()

    if close_current_project:
        current_project = project_manager.GetCurrentProject()
        project_manager.CloseProject(current_project)

    return project_manager


def initialize_project(project_manager, project_name="working_project"):
    """
    * load working project if the project is exist.
    * create working project if the project is not exist.

    Parameters
    ----------
    project_manager : ProjectManager
        a ProjectManager instance
    project_name : str
        a project name.

    Returns
    -------
    Project
        a Project instance
    """
    project = project_manager.LoadProject(project_name)
    if not project:
        print("Unable to loat a project '" + project_name + "'")
        print("Then creating a project '" + project_name + "'")
        project = project_manager.CreateProject(project_name)
        print(f'"{project_name}" is created')
    else:
        print(f'"{project_name}" is loaded')

    return project


def open_page(page_name: str) -> bool:
    result = resolve.OpenPage(page_name)
    return result


def save_project(project_manager):
    ret_val = project_manager.SaveProject()
    if ret_val is not True:
        print("Failed to save project...")
    else:
        print("Project is saved")


def set_project_settings_from_dict(project, params):
    """
    set project settings from the dictionary type parameters.

    Parameters
    ----------
    project : Project
        a Project instance
    parames : dict
        dictionary type parameters
    """
    print("Now this script is setting the project settings...")
    for name, value in params.items():
        result = project.SetSetting(name, value)
        if result:
            print(f'    "{name}" = "{value}" is OK.')
        else:
            print(f'    "{name}" = "{value}" is NGGGGGGGGGGGGGGGGGG.')
        if name == "timelineFrameRate":
            current_page = resolve.GetCurrentPage() 
            result = project.SetRenderSettings({'FrameRate': float(value)})
            if result:
                print(f'    "{name}" = "{value}" is OK in RenderSettings.')
            else:
                print(f'    "{name}" = "{value}" is NGGGGGG in RenderSettings')
            resolve.OpenPage(current_page)
    print("project settings has done")


def create_timeline(timeline_name):
    project = resolve.GetProjectManager().GetCurrentProject()
    media_pool = project.GetMediaPool()
    timeline = media_pool.CreateEmptyTimeline(timeline_name)

    return timeline


def set_current_timeline(timeline):
    project = resolve.GetProjectManager().GetCurrentProject()
    project.SetCurrentTimeline(timeline)


def add_files_to_media_pool(media_path):
    """
    add clips to the media pool.

    Parameters
    ----------
    media_path : str
        A path.
        Both directory and file are permitted (see the examples).

    Examples
    --------
    >>> media_path = Path('D:/abuse/2020/031_cms_for_video_playback/img_seq')
    >>> add_clips_to_media_pool(media_path)

    >>> media_path = Path('D:/abuse/2020/031_cms_for_video_playback/img_seq/hoge.png')
    >>> add_clips_to_media_pool(media_path)
    """
    resolve.OpenPage("media")
    media_storage = resolve.GetMediaStorage()
    clip_list = media_storage.AddItemListToMediaPool(media_path)

    return clip_list


def get_media_pool_clip_list_and_clip_name_list(project):
    """
    Parametes
    ---------
    project : Project
        a Project instance

    Returns
    -------
    clip_return_list : list
        clip list
    clip_name_list : list
        clip name list

    Examples
    --------
    >>> resolve, project_manager = init_resolve(
    ...     close_current_project=close_current_project)
    >>> get_media_pool_clip_dict_list(project)
    [<PyRemoteObject object at 0x000001BB29001630>,
     <PyRemoteObject object at 0x000001BB290016D8>,
     <PyRemoteObject object at 0x000001BB29001720>]
    ['dst_grad_tp_1920x1080_b-size_64_[0001-0024].png',
     'src_grad_tp_1920x1080_b-size_64_[0000-0023].png',
     'dst_grad_tp_1920x1080_b-size_64_resolve_[0000-0023].tif']
    """
    media_pool = project.GetMediaPool()
    root_folder = media_pool.GetRootFolder()
    clip_list = root_folder.GetClipList()
    clip_name_list = []
    clip_list_for_return = []
    for clip in clip_list:
        clip_name_list.append(clip.GetName())
        clip_list_for_return.append(clip)

    return clip_list_for_return, clip_name_list


def create_timeline_from_clip(
        clip_list, timeline_name="dummy"):
    """
    Parametes
    ---------
    clip_list : list
        list of clip
    timeline_name : str
        timeline name
    """
    resolve.OpenPage("edit")
    media_pool = resolve.GetProjectManager().GetCurrentProject().GetMediaPool()
    timeline = media_pool.CreateTimelineFromClips(timeline_name, clip_list)

    return timeline


def add_clips_to_the_current_timeline(clip_list):
    """
    Parametes
    ---------
    clip_list : list
        list of clip
    """
    resolve.OpenPage("edit")
    project = resolve.GetProjectManager().GetCurrentProject()
    media_pool = project.GetMediaPool()
    timeline_item = media_pool.AppendToTimeline(clip_list)

    return timeline_item


def eliminate_frame_idx_from_clip_name(clip_name):
    """
    Examples
    --------
    >>> clip_name = 'dst_grad_tp_1920x1080_b-size_64_[0001-0024].png'
    >>> eliminate_frame_idx_from_clip_name(clip_name)
    eliminate_frame_idx_from_clip_name(clip_name)
    """
    eliminated_name = re.sub('_\[\d+-\d+\]', '', clip_name)

    return eliminated_name


def eliminate_frame_idx_and_ext_from_clip_name(clip_name):
    """
    Examples
    --------
    >>> clip_name = 'dst_grad_tp_1920x1080_b-size_64_[0001-0024].png'
    >>> eliminate_frame_idx_from_clip_name(clip_name)
    eliminate_frame_idx_from_clip_name(clip_name)
    """
    eliminated_name = re.sub('_\[\d+-\d+\]\..+$', '', clip_name)

    return eliminated_name


def load_encode_preset(project, preset_name):
    result = project.LoadRenderPreset(preset_name)
    if result:
        print(f'preset "{preset_name}" is loaded.')
    else:
        print(f'INVALID PRESET "{preset_name}" NGGGGGGGGGGGGGGGGGGGGGGG.')


def set_render_format_codec_settings(
        project, format_str='mov', codec='ProRes422HQ'):
    """
    Parameters
    ----------
    project : Project
        a Project instance
    """
    result = project.SetCurrentRenderFormatAndCodec(format_str, codec)
    if result:
        print(f"format={format_str}, codec={codec} is OK")
    else:
        print(f"format={format_str}, codec={codec} is NGGGGGGGGGGGGGG")


def set_render_settings(project, out_path):
    """
    Parameters
    ----------
    project : Project
        a Project instance
    out_path : Path
        output file name
    """
    target_dir = str(out_path.parent)
    name_prefix = str(out_path.name)

    result = project.SetRenderSettings(
        {"TargetDir": target_dir, "CustomName": name_prefix})
    if result:
        print(f"TargetDir={target_dir}, CustomName={name_prefix} is OK")
    else:
        print(f"TargetDir={target_dir}, CustomName={name_prefix} is NGGGGGGGG")


def encode(resolve, project, out_path, format_str, codec, preset_name=None):
    """
    Parameters
    ----------
    resolve : Resolve
        a Resolve instance
    project : Project
        a Project instance
    out_path : Path
        output file name
    format_str : str
        output format. ex. "mp4", "mov", etc...
    codec : str
        codec. ex "H264_NVIDIA", "ProRes422HQ", etc...
    preset_name : str
        render preset name
    """
    resolve.OpenPage("deliver")

    project.DeleteAllRenderJobs()

    if preset_name is not None:
        load_encode_preset(project, preset_name)
    set_render_format_codec_settings(project, format_str, codec)
    set_render_settings(project, out_path)

    project.AddRenderJob()
    project.StartRendering()
    project.DeleteAllRenderJobs()
    wait_for_rendering_completion(project)


def _debug_save_dict_as_txt(fname, data):
    current_directory = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    buf = []
    with open(fname, 'wt') as f:
        for key, value in data.items():
            text_data = f"{key}: {value}"
            print(text_data)
            buf.append(text_data)
        f.write("\n".join(buf))
    os.chdir(current_directory)


def _debug_print_and_save_encode_settings(project):
    current_directory = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    format_list = project.GetRenderFormats()
    buf = ""
    for render_format_name, ext in format_list.items():
        codecs = project.GetRenderCodecs(ext)
        buf += f"=== {ext} ===\n"
        for key, value in codecs.items():
            buf += f"{key}: {value}\n"
        buf += "\n"
        print(f"=== {ext} ===")
        print(codecs)
        print('')
    with open("./dv17_codecs.txt", 'wt') as f:
        f.write(buf)
    os.chdir(current_directory)


def _debug_print_and_save_project_settings(project):
    project_settings = project.GetSetting()
    pprint.pprint(project_settings)
    _debug_save_dict_as_txt("./project_settings_list.txt", project_settings)


def _debug_print_project_settings(project, key_list=None):
    project_settings = project.GetSetting()

    if key_list is None:
        pprint.pprint(project_settings)
    else:
        for key in key_list:
            print(f"{key} = ", end="")
            pprint.pprint(project_settings[key])


def test_func(close_current_project=True):
    """
    test
    """
    # parameter definition
    out_path = Path(
        'D:/abuse/2020/031_cms_for_video_playback/mp4_to_png/python_test_')
    format_str = "mp4"
    codec = "H264_NVIDIA"
    preset_name = "H264_lossless"
    project_params = dict(
        timelineResolutionHeight="1080",
        timelineResolutionWidth="1920",
        videoMonitorFormat="HD 1080p 24",
        timelineFrameRate="24.000",
        timelinePlaybackFrameRate="24",
        colorScienceMode="davinciYRGBColorManagedv2",
        rcmPresetMode="Custom",
        separateColorSpaceAndGamma="1",
        colorSpaceInput="Rec.709",
        colorSpaceInputGamma="Gamma 2.4",
        colorSpaceTimeline="Rec.709",
        colorSpaceTimelineGamma="Gamma 2.4",
        colorSpaceOutput="Rec.709",
        colorSpaceOutputGamma="Gamma 2.4",
        inputDRT="None",
        outputDRT="None",
    )

    # set
    project_manager = init_resolve(
        close_current_project=close_current_project)
    project = initialize_project(project_manager)

    set_project_settings_from_dict(project, project_params)

    media_path = Path('D:/abuse/2020/031_cms_for_video_playback/img_seq')
    clip_list = add_files_to_media_pool(media_path=media_path)
    timeline = create_timeline(timeline_name="hogefuga")
    set_current_timeline(timeline=timeline)
    add_clips_to_the_current_timeline(clip_list=clip_list)

    encode(resolve, project, out_path, format_str, codec, preset_name)

    return project


def get_avilable_parameters(project_name="working_project"):
    project_manager = init_resolve(
        close_current_project=True)
    project = initialize_project(
        project_manager=project_manager, project_name=project_name)
    _debug_print_and_save_project_settings(project)
    _debug_print_and_save_encode_settings(project)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # get_avilable_parameters(project_name="aaa")

    project_manager = init_resolve(close_current_project=False)
    project = project_manager.GetCurrentProject()
    # key_list = [
    #     "separateColorSpaceAndGamma", "colorScienceMode",
    #     "colorSpaceOutput", "colorSpaceOutputGamma", "colorAcesODT"
    # ]
    key_list = [
        "timelineFrameRate", 'timelinePlaybackFrameRate'
    ]
    _debug_print_project_settings(project, key_list=key_list)
