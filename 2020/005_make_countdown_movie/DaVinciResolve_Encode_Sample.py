# -*- coding: utf-8 -*-

"""
Example DaVinci Resolve script
==============================

Add this script's directory to PYTHONPATH
so it can be detected by DaVinci Resolve.

Examples
--------
import imp
import sample_encode as se
imp.reload(se)
se.main_func()
"""

import DaVinciResolveScript as drs
from collections import OrderedDict
import pprint
from itertools import product
import time


MEDIA_POOL_DIR_LIST = [
    "D:\\abuse\\2020\\005_make_countdown_movie\\movie_seq",
    "C:\\Users\\toruv\\OneDrive\\work\\sample_code\\2020\\005_make_countdown_movie\\wav"
]

RENDER_OUT_DIR = "D:\\Resolve\\render_out"
REVISION = "05"

RESOLUTION_LIST = [(1920, 1080), (3840, 2160)]
FRAMERATE_LIST = [24, 30, 60]
DYNAMIC_RANGE_LIST = ["SDR", "HDR"]


PROJECT_SETTINGS_UHD_60P_HDR = OrderedDict(
    timelineResolutionWidth=3840,
    timelineResolutionHeight=2160,
    timelineFrameRate=60,
    dynamicRange="HDR",
    revision="05",
)


def _debug_make_timeline_from_media_storage(
        project, media_storage):
    media_storage.AddItemListToMediaPool(MEDIA_POOL_DIR_LIST)
    media_pool = project.GetMediaPool()
    folder = media_pool.GetRootFolder()
    clip_list = folder.GetClipList()

    for clip in clip_list:
        print(f"clip = {clip.GetName()}")

    # とりあえず最後のクリップをタイムラインに追加する
    clip = clip_list[7]
    audio_clip = clip_list[0]
    time_line_name = "timeline"
    time_line = project.GetTimelineByIndex(1)
    media_pool.AppendToTimeline(clip)


def _debug_set_project_settings(project, project_settings):
        
    for name, value in project_settings.items():
        result = project.SetSetting(name, value)
        if result:
            print(f'"{name}" = "{value}" is OK.')
        else:
            print(f'"{name}" = "{value}" is NGGGGGGGGGGGGGGGGGG.')

    # key = "hdrDolbyMasterDisplay"
    # value = project.GetSetting(key)
    # print(f"the type of {value} is {type(value)}")
    # result = project.SetSetting(key, value)
    # print(f"result = {result}")
    # pprint.pprint(project.GetSetting())


def main_func():
    resolve = drs.scriptapp("Resolve")
    fusion = resolve.Fusion()

    for resolution, framerate, dynamic_range in product(
            RESOLUTION_LIST, FRAMERATE_LIST, DYNAMIC_RANGE_LIST):
        project_settings = dict(
            timelineResolutionWidth=resolution[0],
            timelineResolutionHeight=resolution[1],
            timelineFrameRate=framerate,
            dynamicRange=dynamic_range,
            revision=REVISION
        )
        encode_specific_format(
            resolve=resolve, project_settings=project_settings)


def change_project_settings(project, project_settings):
    frame_rate = project_settings["timelineFrameRate"]
    width = project_settings["timelineResolutionWidth"]
    height = project_settings["timelineResolutionHeight"]

    result = True
    if not project.SetSetting("timelineFrameRate", str(frame_rate)):
        result = False
    if not project.SetSetting("timelineResolutionWidth", str(width)):
        result = False
    if not project.SetSetting("timelineResolutionHeight", str(height)):
        reslt = False

    if not result:
        print("======================================")
        print("Error! project settings is invalid.")
        print(project_settings)
        print("======================================")
        print("")

    # for name, value in project_settings.items():
    #     result = project.SetSetting(name, value)
    #     if result:
    #         print(f'"{name}" = "{value}" is OK.')
    #     else:
    #         print("========================================")
    #         print("Error")
    #         print(f'"{name}": "{value}" is invalid')


def make_timeline(project, project_settings, media_storage):
    media_storage.AddItemListToMediaPool(MEDIA_POOL_DIR_LIST)
    media_pool = project.GetMediaPool()
    folder = media_pool.GetRootFolder()
    clip_list = folder.GetClipList()

    # time_line = media_pool.CreateEmptyTimeline("zzz_timeline")
    # for clip in clip_list:
    #     clip_name = clip.GetName()
    #     if clip_name.endswith(".wav"):
    #         media_pool.AppendToTimeline(clip)
    #         print(f'"{clip_name}" has added to timeline.')
    #         break

    for clip in clip_list:
        clip_name = clip.GetName()
        dynamic_range = project_settings["dynamicRange"]
        frame_rate = project_settings["timelineFrameRate"]
        width = project_settings["timelineResolutionWidth"]
        height = project_settings["timelineResolutionHeight"]
        compare_str = f"movie_{dynamic_range}_{width}x{height}_{frame_rate}fps"
        if clip_name.startswith(compare_str):
            media_pool.AppendToTimeline(clip)
            print(f'"{clip_name}" has added to timeline.')
            break


def encode(resolve, project, project_settings):
    project.DeleteAllRenderJobs()
    format_str = "mov"
    codec = "DNxHR444_12"
    resolve.OpenPage("deliver")
    # pprint.pprint(project.GetRenderFormats())
    # pprint.pprint(project.GetRenderCodecs(format_str))
    # pprint.pprint(project.GetCurrentRenderFormatAndCodec())
    result = project.SetCurrentRenderFormatAndCodec(format_str, codec)

    dynamic_range = project_settings["dynamicRange"]
    frame_rate = project_settings["timelineFrameRate"]
    width = project_settings["timelineResolutionWidth"]
    height = project_settings["timelineResolutionHeight"]    
    revision = project_settings["revision"]
    outname = f"Countdown_{dynamic_range}_{width}x{height}_"\
        + f"{frame_rate}P_master"

    if not result:
        print("Error! codec settings is invalid")
        print(f"    format={format_str}", codec={codec})
    result = project.SetRenderSettings(
        {"TargetDir" : RENDER_OUT_DIR, "CustomName" : outname})
    if not result:
        print("Error! RenderSetting is invalid")
    project.AddRenderJob()
    project.StartRendering()
    project.DeleteAllRenderJobs()


def get_base_project_name(project_settings):
    if project_settings["dynamicRange"] == "HDR":
        tf = "ST2084"
        gamut = "BT2020"
    elif project_settings["dynamicRange"] == "SDR":
        tf = "Gamma2.4"
        gamut = "BT709"

    frame_rate = project_settings["timelineFrameRate"]
    return f"{tf}-{gamut}_{frame_rate}P"


def wait_for_rendering_completion(project):
    while project.IsRenderingInProgress():
        time.sleep(1)
    return


def encode_specific_format(
        resolve, project_settings=PROJECT_SETTINGS_UHD_60P_HDR):
    base_project_name = get_base_project_name(project_settings)

    project_manager = resolve.GetProjectManager()
    project_manager.OpenFolder("Countdown")

    # force close base project if it is opened.
    # project = project_manager.LoadProject(base_project_name)
    # project_manager.CloseProject(project)

    # load project
    project = project_manager.LoadProject(base_project_name)
    change_project_settings(project, project_settings)

    media_storage = resolve.GetMediaStorage()
    make_timeline(project, project_settings, media_storage)

    encode(
        resolve=resolve, project=project, project_settings=project_settings)

    wait_for_rendering_completion(project)

    project_manager.CloseProject(project)
    time.sleep(1)


if __name__ == '__main__':
    main_func()
