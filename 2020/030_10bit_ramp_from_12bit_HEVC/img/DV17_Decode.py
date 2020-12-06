# -*- coding: utf-8 -*-
"""
Title
==============

Description.

"""

# import standard libraries
import os
import sys
from pathlib import PureWindowsPath, WindowsPath
import time
import pprint

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

MAIN_PRJ_NAME = "decode_H265_12bit"
DUMMY_PRJ_NAME = "dummy_prj"
MP4_DIR = "D:/abuse/2020/030_10bit_ramp_from_12bit_HEVC/mp4"
RENDER_OUT_DIR = "D:/abuse/2020/030_10bit_ramp_from_12bit_HEVC/tiff"

"""
import imp
import DV17_Decode as dv17
imp.reload(dv17)
dv17.main_func2()
"""

def save_dict_as_txt(fname, data):
    buf = []
    with open(fname, 'wt') as f:
        for key, value in data.items():
            text_data = f"{key}: {value}"
            print(text_data)
            buf.append(text_data)
        f.write("\n".join(buf))


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


def get_max_value_from_clip_name(clip_name):
    max_value = clip_name[-8:-4]
    return max_value


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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
