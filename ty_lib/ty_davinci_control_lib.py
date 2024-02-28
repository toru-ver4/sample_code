# -*- coding: utf-8 -*-
"""
A library for DaVinci Resolve control
=====================================
"""

# import standard libraries
import os
import sys
import time
import pprint
from pathlib import Path

# import third-party libraries
from python_get_resolve import GetResolve

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

resolve = GetResolve()

###################################
# Page
###################################
MEDIA_PAGE_STR = "media"
CUT_PAGE_STR = "cut"
EDIT_PAGE_STR = "edit"
FUSION_PAGE_STR = 'fusion'
COLOR_PAGE_STR = "color"
DELIVER_PAGE_STR = "deliver"

###################################
# Project Settings
###################################
PRJ_PARAM_NONE = "None"
PRJ_PARAM_ENABLE = "1"
PRJ_PARAM_DISABLE = "1"

PRJ_TIMELINE_RESOLUTION_1920 = "1920"
PRJ_TIMELINE_RESOLUTION_1080 = "1080"
PRJ_TIMELINE_RESOLUTION_3840 = "3840"
PRJ_TIMELINE_RESOLUTION_2160 = "2160"

PRJ_SDI_SINGLE_LINK = "single_link"

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
PRJ_GAMMA_STR_GAMMA26 = "Gamma 2.6"

PRJ_COLOR_SPACE_REC2020 = "Rec.2020"
PRJ_COLOR_SPACE_REC709 = "Rec.709"
PRJ_COLOR_SPACE_P3D65 = "P3-D65"

PRJ_ACES_ODT_P3D65_PQ_108 = "P3-D65 ST2084 (108 nits)"
PRJ_ACES_ODT_P3D65_PQ_1000 = "P3-D65 ST2084 (1000 nits)"
PRJ_ACES_ODT_P3D65_PQ_4000 = "P3-D65 ST2084 (4000 nits)"

PRJ_ACES_ODT_REC2020_PQ_1000 = "Rec.2020 ST2084 (1000 nits)"

PRJ_LUMINANCE_MODE_CUSTOM = "Custom"
PRJ_WORKING_LUMINANCE_MAX = "10000"

###################################
# File Extenstion
###################################
OUT_FILE_EXTENSTION_MP4 = "mp4"
OUT_FILE_EXTENSTION_MOV = "mov"
OUT_FILE_EXTENSTION_EXR = "exr"
OUT_FILE_EXTENSTION_DPX = "dpx"
OUT_FILE_EXTENSTION_DPX = "tif"

###################################
# Codec + Encoder
###################################
CODEC_APPLE_PRORES_422 = "ProRes422"
CODEC_APPLE_PRORES_422_HQ = "ProRes422HQ"
CODEC_APPLE_PRORES_422_LT = "ProRes422LT"
CODEC_APPLE_PRORES_422_PROXY = "ProRes422P"
CODEC_APPLE_PRORES_4444 = "ProRes4444"
CODEC_APPLE_PRORES_4444_XQ = "ProRes4444XQ"

CODEC_DN_X_HR_444_10_BIT = "DNxHR444_10"
CODEC_DN_X_HR_444_12_BIT = "DNxHR444_12"
CODEC_DN_X_HR_HQ = "DNxHRHQ"
CODEC_DN_X_HR_HQX_10_BIT = "DNxHRHQX_10"
CODEC_DN_X_HR_HQX_12_BIT = "DNxHRHQX_12"

CODEC_H264 = "H264"
CODEC_H264_NVIDIA = "H264_NVIDIA"
CODEC_H265 = "H265"
CODEC_H265_NVIDIA = "H265_NVIDIA"

CODEC_TIF_RGB_16_BITS = "RGB16"
CODEC_TIF_RGB_16_BITS_LZW = "RGB16LZW"
CODEC_TIF_RGB_8_BITS = "RGB8"
CODEC_TIF_RGB_8_BITS_LZW = "RGB8LZW"
CODEC_TIF_XYZ_16_BITS = "XYZ16"
CODEC_TIF_XYZ_16_BITS_LZW = "XYZ16LZW"

CODEC_DPX_RGB_10_BITS = "RGB10"
CODEC_DPX_RGB_12_BITS = "RGB12"
CODEC_DPX_RGB_16_BITS = "RGB16"

CODEC_EXR_RGB_FLOAT = "RGBFloat"
CODEC_EXR_RGB_FLOAT_DWAA = "RGBFloatDWAA"
CODEC_EXR_RGB_FLOAT_DWAB = "RGBFloatDWAB"
CODEC_EXR_RGB_FLOAT_PIZ = "RGBFloatPIZ"
CODEC_EXR_RGB_FLOAT_RLE = "RGBFloatRLE"
CODEC_EXR_RGB_FLOAT_ZIP = "RGBFloatZIP"
CODEC_EXR_RGB_HALF = "RGBHalf"
CODEC_EXR_RGB_HALF_DWAA = "RGBHalfDWAA"
CODEC_EXR_RGB_HALF_DWAB = "RGBHalfDWAB"
CODEC_EXR_RGB_HALF_PIZ = "RGBHalfPIZ"
CODEC_EXR_RGB_HALF_RLE = "RGBHalfRLE"
CODEC_EXR_RGB_HALF_ZIP = "RGBHalfZIP"

PRJECT_SETTINGS_SAMPLE_BT2100 = dict(
    timelineResolutionWidth=PRJ_TIMELINE_RESOLUTION_1920,
    timelineResolutionHeight=PRJ_TIMELINE_RESOLUTION_1080,
    timelinePlaybackFrameRate=PRJ_TIMELINE_PLAYBACK_FRAMERATE_24,
    videoMonitorFormat=PRJ_VIDEO_MONITOR_FORMAT_HD_1080P24FPS,
    timelineFrameRate=PRJ_TIMELINE_FRAMERATE_24,
    videoMonitorUse444SDI=PRJ_PARAM_DISABLE,
    videoMonitorSDIConfiguration=PRJ_SDI_SINGLE_LINK,
    videoDataLevels=PRJ_VIDEO_DATA_LEVEL_LIMITED,
    videoMonitorUseHDROverHDMI=PRJ_PARAM_ENABLE,
    colorScienceMode=PRJ_COLOR_SCIENCE_MODE_RCM_ON,
    rcmPresetMode=PRJ_PRESET_MODE_CUSTOM,
    separateColorSpaceAndGamma=PRJ_PARAM_ENABLE,
    colorSpaceInput=PRJ_COLOR_SPACE_REC2020,
    colorSpaceInputGamma=PRJ_GAMMA_STR_ST2084,
    colorSpaceTimeline=PRJ_COLOR_SPACE_REC2020,
    colorSpaceTimelineGamma=PRJ_GAMMA_STR_ST2084,
    colorSpaceOutput=PRJ_COLOR_SPACE_REC2020,
    colorSpaceOutputGamma=PRJ_GAMMA_STR_ST2084,
    timelineWorkingLuminance=PRJ_WORKING_LUMINANCE_MAX,
    timelineWorkingLuminanceMode=PRJ_LUMINANCE_MODE_CUSTOM,
    inputDRT=PRJ_PARAM_NONE,
    outputDRT=PRJ_PARAM_NONE,
    hdrMasteringLuminanceMax="1000",
    hdrMasteringOn=PRJ_PARAM_ENABLE
)


def wait_for_rendering_completion(project):
    while project.IsRenderingInProgress():
        time.sleep(1)
    return


def close_and_remove_project(project_name):
    """
    Delete the project.
    """
    project_manager = resolve.GetProjectManager()
    project_manager.CloseProject(project_manager.GetCurrentProject())
    project_manager.DeleteProject(project_name)


def close_current_project():
    project_manager = resolve.GetProjectManager()
    project_manager.CloseProject(project_manager.GetCurrentProject())


def get_project_manager(close_current_project=False):
    """
    * Create ProjectManager instance
    * Close current project for initialize if needed.

    Returns
    -------
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
    * Load working project if the project is exist.
    * Create working project if the project is not exist.

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


def archive_project(project_name="hoge", archive_path="./hoge.dra"):
    close_current_project()
    project_manager = get_project_manager()
    print(f"project_name = {project_name}, path = {archive_path}")
    result = project_manager.ArchiveProject(
        project_name, archive_path, True, False, False)
    if result is True:
        print("ArchiveProject is succeeded")
    else:
        print("ArchiveProject is filed...")
    """
    ArchiveProject(
        projectName, filePath, isArchiveSrcMedia=True,
        isArchiveRenderCache=True, isArchiveProxyMedia=False) 
    """


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


def set_cuurent_timecode(timeline, timecode):
    """
    Parameters
    ----------
    timeline : Timeline
    timecode : str
        eg. "01:00:00:12", "01:00:01:00"
    """
    timeline.SetCurrentTimecode(timecode)


def create_timeline(timeline_name):
    """
    Create empty timeline
    """
    project = resolve.GetProjectManager().GetCurrentProject()
    media_pool = project.GetMediaPool()
    timeline = media_pool.CreateEmptyTimeline(timeline_name)

    if timeline is not None:
        print(f"{timeline_name} timeline was created")
    else:
        print(f"[Error!] Failed to create {timeline_name} timeline")
        print("Maybe the timeline that has samename is exist")
        sys.exit(1)

    return timeline


def set_current_timeline(timeline):
    project = resolve.GetProjectManager().GetCurrentProject()
    project.SetCurrentTimeline(timeline)


def remove_all_timeline(project):
    """
    Remove all timeline.
    """
    num_of_timeline = project.GetTimelineCount()
    print(f"num_of_timeline = {num_of_timeline}")
    media_pool = project.GetMediaPool()

    timelines = []

    for idx in range(num_of_timeline):
        # timeline index starts from 1 not 0.
        timeline = project.GetTimelineByIndex(idx + 1)
        timelines.append(timeline)

    if timelines != []:
        media_pool.DeleteTimelines(timelines)


def add_files_to_media_pool(media_path):
    """
    add clips to the media pool.

    Parameters
    ----------
    media_path : str
        A path.
        Both directory and file are permitted (see the examples).

    Returns
    -------
    Array
        An array of the Meida Clip

    Examples
    --------
    >>> media_path = str(Path('D:/abuse/2020/031_cms_for_video_playback/img_seq'))
    >>> add_files_to_media_pool(media_path)
    [<BlackmagicFusion.PyRemoteObject object at 0x000002A68DC66130>,
     <BlackmagicFusion.PyRemoteObject object at 0x000002A68DC66110>,
     <BlackmagicFusion.PyRemoteObject object at 0x000002A68DC660F0>,
     <BlackmagicFusion.PyRemoteObject object at 0x000002A68DC654D0>,
     <BlackmagicFusion.PyRemoteObject object at 0x000002A68DC654B0>,
     <BlackmagicFusion.PyRemoteObject object at 0x000002A68DC660D0>,
     <BlackmagicFusion.PyRemoteObject object at 0x000002A68DC660B0>,
     <BlackmagicFusion.PyRemoteObject object at 0x000002A68DC65490>]
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
    clip_list : list
        clip list
    clip_name_list : list
        clip name list

    Examples
    --------
    >>> project = resolve.GetProjectManager().GetCurrentProject()
    >>> clip_list, clip_name_list\
    ...     = get_media_pool_clip_list_and_clip_name_list(project)
    >>> print(clip_list)
    [<BlackmagicFusion.PyRemoteObject object at 0x000001DD451F54D0>,
     <BlackmagicFusion.PyRemoteObject object at 0x000001DD451F54B0>,
     <BlackmagicFusion.PyRemoteObject object at 0x000001DD451F60D0>,
     <BlackmagicFusion.PyRemoteObject object at 0x000001DD451F60B0>]
    >>> print(clip_name_list)
    ['countdown_SDR_1920x1080_24fps_Rev9_hevc_yuv420p10le_qp-0.mov',
     'countdown_HDR_1920x1080_24fps_Rev9_hevc_yuv420p10le_qp-0.mov',
     'Sample Timeline',
     'SDR_TyTP_P3D65_Mhi_5.62_REF_WHITE_203_[0000-0179].png']
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
    out_path : pathlib.Path or str
        output file name
    """
    target_dir = str(Path(out_path).parent)
    name_prefix = str(Path(out_path).name)

    result = project.SetRenderSettings(
        {"TargetDir": target_dir, "CustomName": name_prefix})
    if result:
        print(f"TargetDir={target_dir}, CustomName={name_prefix} is OK")
    else:
        print(f"TargetDir={target_dir}, CustomName={name_prefix} is NGGGGGGGG")


def encode(project, out_path, format_str, codec, preset_name=None):
    """
    Parameters
    ----------
    project : Project
        a Project instance
    out_path : pathlib.Path or str
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
    else:
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
    with open("./resolve_codecs.txt", 'wt') as f:
        f.write(buf)
    os.chdir(current_directory)


def _debug_print_and_save_project_settings(project):
    project_settings = project.GetSetting()
    pprint.pprint(project_settings)
    _debug_save_dict_as_txt("./project_settings_list.txt", project_settings)


def _debug_print_and_save_timeline_settings(project):
    timeline = create_timeline(timeline_name="dummy")
    timeline_settings = timeline.GetSetting()
    pprint.pprint(timeline_settings)
    _debug_save_dict_as_txt("./timeline_settings_list.txt", timeline_settings)


def _debug_print_project_settings(project, key_list=None):
    project_settings = project.GetSetting()

    if key_list is None:
        pprint.pprint(project_settings)
    else:
        for key in key_list:
            print(f"{key} = ", end="")
            pprint.pprint(project_settings[key])


def sample_func():
    """
    test
    """
    project_name = "Python Script Test"
    timeline_name = "Sample Timeline"

    #################################################
    # Create Project
    #################################################
    close_and_remove_project(project_name=project_name)
    project_manager = get_project_manager()
    project = initialize_project(
        project_manager=project_manager, project_name=project_name
    )
    open_page(EDIT_PAGE_STR)
    remove_all_timeline(project=project)  # just in case
    set_project_settings_from_dict(
        project=project, params=PRJECT_SETTINGS_SAMPLE_BT2100
    )

    #################################################
    # Add Clips
    #################################################
    src_media_path_list = [
        "./videos/countdown_SDR_1920x1080_24fps_Rev9_hevc_yuv420p10le_qp-0.mov",
        "./videos/countdown_HDR_1920x1080_24fps_Rev9_hevc_yuv420p10le_qp-0.mov"
    ]
    # convert to relative path to absolute path using Path module
    src_media_path_list = [
        str(Path(src_media_path).resolve())
        for src_media_path in src_media_path_list
    ]
    clip_list = add_files_to_media_pool(media_path=src_media_path_list)
    timeline = create_timeline_from_clip(
        clip_list=clip_list, timeline_name=timeline_name
    )

    # If you want to change the playback position,
    # you can use `set_cuurent_timecode`
    set_cuurent_timecode(
        timeline=timeline, timecode="01:00:01:00"
    )

    #################################################
    # Encode
    #################################################
    out_path = str(Path("./videos/sample_out.mp4").resolve())
    format_str = OUT_FILE_EXTENSTION_MP4
    codec = CODEC_H264
    preset_name = None
    encode(project, out_path, format_str, codec, preset_name=preset_name)

    out_path = str(Path("./videos/h265_main10_444_qp0.mp4").resolve())
    format_str = None
    codec = None
    preset_name = "H265_main10_444_qp0_preset"
    encode(project, out_path, format_str, codec, preset_name=preset_name)

    #################################################
    # Save
    #################################################
    save_project(project_manager=project_manager)


def get_avilable_parameters(project_name="sample_project"):
    """
    Dump all parameters for "Project:SetSetting", "Timeline:SetSetting" and
    "Project:SetCurrentRenderFormatAndCodec".
    """
    project_manager = get_project_manager(
        close_current_project=True)
    project = initialize_project(
        project_manager=project_manager, project_name=project_name)
    _debug_print_and_save_project_settings(project)
    _debug_print_and_save_timeline_settings(project)
    _debug_print_and_save_encode_settings(project)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sample_func()
    # get_avilable_parameters(project_name="aaa")
