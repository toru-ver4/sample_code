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
dv_lib.test_func(close_current_project=False)

Example 2
---------
import imp
import ty_davinci_control_lib as dv_lib
imp.reload(dv_lib)
project = dv_lib.test_func()
...
dv_lib._debug_print_and_save_project_settings(project)
"""


def set_project_settings(project):
    project.SetSetting("timelineOutputResolutionHeight", "720")
    project.SetSetting("timelineOutputResolutionWidth", "1280")
    project.SetSetting("timelineResolutionHeight", "720")
    project.SetSetting("timelineResolutionWidth", "1280")
    project.SetSetting("videoMonitorFormat", "HD 720p 24")
    project.SetSetting("timelineFrameRate", "24.000")
    project.SetSetting("timelinePlaybackFrameRate", "24")  # not working
    project.SetSetting("colorScienceMode", "davinciYRGBColorManagedv2")
    project.SetSetting("rcmPresetMode", "Custom")
    project.SetSetting("separateColorSpaceAndGamma", "1")
    # project.SetSetting("", "")
    project.SetSetting("colorSpaceInput", "Rec.709")
    project.SetSetting("colorSpaceInputGamma", "Gamma 2.4")
    project.SetSetting("colorSpaceTimeline", "Rec.709")
    project.SetSetting("colorSpaceTimelineGamma", "Gamma 2.4")
    project.SetSetting("colorSpaceOutput", "Rec.709")
    project.SetSetting("colorSpaceOutputGamma", "Gamma 2.4")
    project.SetSetting("inputDRT", "None")
    project.SetSetting("outputDRT", "None")


def add_clip(resolve, project):
    set_project_settings(project)
    resolve.OpenPage("media")
    media_storage = resolve.GetMediaStorage()
    media_pool = project.GetMediaPool()
    media_path = str(PureWindowsPath(MP4_DIR))
    media_storage.AddItemListToMediaPool(media_path)


def encode_each_clip(resolve, project, clip_idx):
    resolve.OpenPage("edit")
    media_pool = project.GetMediaPool()
    media_storage = resolve.GetMediaStorage()
    media_path = str(PureWindowsPath(MP4_DIR))
    file_list = media_storage.GetFileList(media_path)
    folder = media_pool.GetCurrentFolder()

    file_name = file_list[clip_idx]
    max_val = get_max_value_from_clip_name(file_name)

    timeline = media_pool.CreateTimelineFromClips(
        f"{max_val}", folder.GetClipList()[clip_idx])

    project.DeleteAllRenderJobs()
    format_str = "tif"
    codec = "RGB16LZW"
    resolve.OpenPage("deliver")

    # pprint.pprint(project.GetRenderFormats())
    # pprint.pprint(project.GetRenderCodecs("tif"))
    # pprint.pprint(project.GetCurrentRenderFormatAndCodec())

    result = project.SetCurrentRenderFormatAndCodec(format_str, codec)

    outname = f"{max_val}"

    if not result:
        print("Error! codec settings is invalid")
        print(f"    format={format_str}", codec={codec})
    result = project.SetRenderSettings(
        {"TargetDir": str(PureWindowsPath(RENDER_OUT_DIR)),
         "CustomName": outname})
    if not result:
        print("Error! RenderSetting is invalid")
    project.AddRenderJob()
    project.StartRendering()
    project.DeleteAllRenderJobs()


def wait_for_rendering_completion(project):
    while project.IsRenderingInProgress():
        time.sleep(1)
    return


def main_func2():
    # load project
    resolve = dvr_script.scriptapp("Resolve")
    fusion = resolve.Fusion()
    projectManager = resolve.GetProjectManager()
    projectManager.DeleteProject(MAIN_PRJ_NAME)
    project = projectManager.CreateProject(MAIN_PRJ_NAME)
    if not project:
        print("Unable to loat a project '" + MAIN_PRJ_NAME + "'")
        sys.exit()
    add_clip(resolve, project)
    projectManager.SaveProject()
    projectManager.CloseProject(project)
    time.sleep(1)

    clip_num = len(list(WindowsPath(MP4_DIR).glob('*.mp4')))

    for clip_idx in range(clip_num):
        project = projectManager.LoadProject(MAIN_PRJ_NAME)
        encode_each_clip(resolve, project, clip_idx)
        wait_for_rendering_completion(project)
        projectManager.CloseProject(project)
        time.sleep(1)


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
    fusion = resolve.Fusion()
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


def get_media_pool_clip_dict_list(project):
    """
    Parametes
    ---------
    project : Project
        a Project instance

    Returns
    -------
    list
        [{clip name1: clip object1}, {clip name2: clip object2}, ...]

    Examples
    --------
    >>> resolve, project_manager = init_davinci17(
    ...     close_current_project=close_current_project)
    >>> get_media_pool_clip_dict_list(project)
    [{'src_grad_tp_1920x1080_b-size_64_[0000-0023].png': <PyRemoteObject object at 0x00000234941E17F8>},
     {'dst_grad_tp_1920x1080_b-size_64_[0001-0024].png': <PyRemoteObject object at 0x00000234941E1798>},
     {'dst_grad_tp_1920x1080_b-size_64_resolve_[0000-0023].tif': <PyRemoteObject object at 0x00000234941E17C8>}]
    """
    media_pool = project.GetMediaPool()
    root_folder = media_pool.GetRootFolder()
    clip_list = root_folder.GetClipList()
    clip_dict_list = []
    for clip in clip_list:
        ddd = {clip.GetName(): clip}
        clip_dict_list.append(ddd)

    return clip_dict_list


def _debug_save_dict_as_txt(fname, data):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    buf = []
    with open(fname, 'wt') as f:
        for key, value in data.items():
            text_data = f"{key}: {value}"
            print(text_data)
            buf.append(text_data)
        f.write("\n".join(buf))
    os.chdir(current_directory)


def _debug_print_and_save_project_settings(project):
    project_settings = project.GetSetting()
    pprint.pprint(project_settings)
    current_directory = os.getcwd()
    _debug_save_dict_as_txt("./project_settings_list.txt", project_settings)


def test_func(close_current_project=True):
    """
    unit test... ?
    """
    resolve, project_manager = init_davinci17(
        close_current_project=close_current_project)
    project = prepare_project(project_manager)

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
    set_project_settings_from_dict(project, project_params)

    media_path = Path('D:/abuse/2020/031_cms_for_video_playback/img_seq')
    add_clips_to_media_pool(resolve, project, media_path)
    clip_dict_list = get_media_pool_clip_dict_list(project)
    print(clip_dict_list)

    return project


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
