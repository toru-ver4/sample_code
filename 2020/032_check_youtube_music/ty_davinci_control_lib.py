# -*- coding: utf-8 -*-
"""
Library for DaVinci17 control
=============================

"""

# import standard libraries
import os
import sys
import time
import pprint
from pathlib import Path
import re

# import third-party libraries
import DaVinciResolveScript as dvr_script

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


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
"""


def wait_for_rendering_completion(project):
    while project.IsRenderingInProgress():
        time.sleep(1)
    return


def init_davinci17(close_current_project=True):
    """
    Initialize davinci17 python environment.

    * create Resolve instance
    * create ProjectManager instance
    * close current project for initialize if needed.

    Returns
    -------
    resolve
        Resolve instance
    project_namager
        ProjectManager instance
    """
    resolve = dvr_script.scriptapp("Resolve")
    # fusion = resolve.Fusion()
    project_manager = resolve.GetProjectManager()

    if close_current_project:
        current_project = project_manager.GetCurrentProject()
        project_manager.CloseProject(current_project)

    return resolve, project_manager


def prepare_project(project_manager, project_name="working_project"):
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
            result = project.SetRenderSettings({'FrameRate': float(value)})
            if result:
                print(f'    "{name}" = "{value}" is OK in RenderSettings.')
            else:
                print(f'    "{name}" = "{value}" is NGGGGGG in RenderSettings')
    print("project settings has done")


def add_clips_to_media_pool(resolve, project, media_path):
    """
    add clips to the media pool.

    Parameters
    ----------
    resolve : Resolve
        a Resolve instance
    project : Project
        a Project instance
    media_path : Path
        a Pathlib instance

    Examples
    --------
    >>> resolve, project_manager = init_davinci17(
    ...     close_current_project=close_current_project)
    >>> project = prepare_project(project_manager)
    >>> media_path = Path('D:/abuse/2020/031_cms_for_video_playback/img_seq')
    >>> add_clips_to_media_pool(resolve, project, media_path)
    """
    resolve.OpenPage("media")
    media_storage = resolve.GetMediaStorage()
    media_pool = project.GetMediaPool()
    media_storage.AddItemListToMediaPool(str(media_path))


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
    >>> resolve, project_manager = init_davinci17(
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
    clip_return_list = []
    for clip in clip_list:
        clip_name_list.append(clip.GetName())
        clip_return_list.append(clip)

    return clip_return_list, clip_name_list


def create_timeline_from_clip(resolve, project, clip_list, timeline_name="dummy"):
    """
    Parametes
    ---------
    resolve : Resolve
        a Resolve iinstance
    project : Project
        a Project instance
    clip_list : list
        list of clip
    timeline_name : str
        timeline name
    """
    resolve.OpenPage("edit")
    media_pool = project.GetMediaPool()
    timeline = media_pool.CreateTimelineFromClips(timeline_name, clip_list)
    # timeline = media_pool.CreateTimelineFromClips(timeline_name, None)

    return timeline


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
    current_directory = os.getcwd()
    _debug_save_dict_as_txt("./project_settings_list.txt", project_settings)


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
    resolve, project_manager = init_davinci17(
        close_current_project=close_current_project)
    project = prepare_project(project_manager)

    set_project_settings_from_dict(project, project_params)

    media_path = Path('D:/abuse/2020/031_cms_for_video_playback/img_seq')
    add_clips_to_media_pool(resolve, project, media_path)
    clip_list, clip_name_list\
        = get_media_pool_clip_list_and_clip_name_list(project)

    timeline = create_timeline_from_clip(
        resolve, project, clip_list, timeline_name="dummy")

    encode(resolve, project, out_path, format_str, codec, preset_name)

    return project


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
