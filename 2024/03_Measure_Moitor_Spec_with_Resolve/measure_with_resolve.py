# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import re
import sys
import shutil
import importlib
import subprocess
from pathlib import Path
from datetime import datetime

# import third-party libraries
from ty_display_pro_hl import read_xyz, save_measure_result

# import my libraries
import ty_davinci_control_lib as dcl
from create_tp_for_measure import create_tp_base_name, create_cv_list

"""
import sys
import importlib
folder_path = 'C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve'
sys.path.append(folder_path)
import measure_with_resolve as mwr
importlib.reload(mwr)
mwr.simple_measure()
"""


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


CCSS_RGBLED = "C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve/ccss/RGBLEDFamily_07Feb11.ccss"


def create_project(project_name="Dummy Project"):
    project_manager = dcl.init_resolve()
    project = dcl.initialize_project(
        project_manager=project_manager, project_name=project_name)
    return project, project_manager


def remove_all_timeline(project):
    num_of_timeline = project.GetTimelineCount()
    print(f"num_of_timeline = {num_of_timeline}")
    media_pool = project.GetMediaPool()

    timelines = []

    for idx in range(num_of_timeline):
        timeline = project.GetTimelineByIndex(idx + 1)
        timelines.append(timeline)

    if timelines != []:
        media_pool.DeleteTimelines(timelines)


def set_project_settings_bt2100(project):
    project_params = dict(
        timelineResolutionWidth=dcl.PRJ_TIMELINE_RESOLUTION_3840,
        timelineResolutionHeight=dcl.PRJ_TIMELINE_RESOLUTION_2160,
        timelinePlaybackFrameRate=dcl.PRJ_TIMELINE_PLAYBACK_FRAMERATE_24,
        videoMonitorFormat=dcl.PRJ_VIDEO_MONITOR_FORMAT_UHD_2160P24FPS,
        timelineFrameRate=dcl.PRJ_TIMELINE_FRAMERATE_24,
        videoDataLevels=dcl.PRJ_VIDEO_DATA_LEVEL_LIMITED,
        videoMonitorUseHDROverHDMI=dcl.PRJ_PARAM_ENABLE,
        colorScienceMode=dcl.PRJ_COLOR_SCIENCE_MODE_RCM_ON,
        rcmPresetMode=dcl.PRJ_PRESET_MODE_CUSTOM,
        separateColorSpaceAndGamma=dcl.PRJ_PARAM_ENABLE,
        colorSpaceInput=dcl.PRJ_COLOR_SPACE_REC2020,
        colorSpaceInputGamma=dcl.PRJ_GAMMA_STR_ST2084,
        colorSpaceTimeline=dcl.PRJ_COLOR_SPACE_REC2020,
        colorSpaceTimelineGamma=dcl.PRJ_GAMMA_STR_ST2084,
        colorSpaceOutput=dcl.PRJ_COLOR_SPACE_REC2020,
        colorSpaceOutputGamma=dcl.PRJ_GAMMA_STR_ST2084,
        inputDRT=dcl.PRJ_PARAM_NONE,
        outputDRT=dcl.PRJ_PARAM_NONE,
    )
    dcl.set_project_settings_from_dict(project=project, params=project_params)


def simple_measure(
        csv_name="./measure_result.csv", ccss_file=None):
    # open dummy project for debug

    project_name = "Measure_AW3225QF"
    dcl.close_and_remove_project(project_name=project_name)
    project, project_manager = create_project(project_name=project_name)
    dcl.open_page(dcl.EDIT_PAGE_STR)
    remove_all_timeline(project=project)
    set_project_settings_bt2100(project=project)

    media_path = Path('C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve/tp_img')
    # dcl.add_clips_to_media_pool(project, media_path)
    # clip_list, clip_name_list\
    #     = dcl.get_media_pool_clip_list_and_clip_name_list(project)
    # print(clip_name_list)

    cv_list = create_cv_list(num_of_block=64)
    color_mask = [1, 1, 1]
    patch_area_ratio = 0.03
    fname_list = [
        create_tp_base_name(
            color_mask=color_mask, cv=cv/1023,
            patch_area_ratio=patch_area_ratio) + ".png"
        for cv in cv_list]
    fname_list = [str(media_path / Path(fname)) for fname in fname_list]

    media_storage = dcl.get_resolve().GetMediaStorage()
    clip_list = media_storage.AddItemListToMediaPool(fname_list)
    for clip in clip_list:
        result = clip.SetClipProperty("Input Color Space", "Rec.2100 ST2084")
        print(f"input color space result = {result}")

    media_pool = project.GetMediaPool()
    tp_timeline = media_pool.CreateTimelineFromClips("TP_Timeline", clip_list)

    print(clip_list[0].GetClipProperty())

    # clips = AddItemListToMediaPool()
    # print(file_list)

    # large_xyz, Yxy = read_xyz(flush=False, ccss_file=ccss_file)
    # ccss_name = Path(ccss_file).stem if ccss_file else "-"
    # save_measure_result(
    #     large_xyz=large_xyz, Yxy=Yxy,
    #     csv_name=csv_name, ccss_name=ccss_name)

    dcl.save_project(project_manager=project_manager)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # simple_measure(csv_name="./measure_result.csv", ccss_file=CCSS_RGBLED)
    project = dcl.get_resolve().GetProjectManager().GetCurrentProject()
    clip_list = project.GetMediaPool().GetRootFolder().GetClipList()
    properties = clip_list[0].GetClipProperty()
    timeline = project.GetCurrentTimeline()
    timeline.SetCurrentTimecode("01:01:00:00")
