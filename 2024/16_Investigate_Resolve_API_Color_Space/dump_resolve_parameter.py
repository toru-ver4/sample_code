# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path
import pprint

# import third-party libraries

# import my libraries
import ty_davinci_control_lib as dcl


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


BASE_PRJ_SETTINGS = dict(
    timelineResolutionWidth=dcl.PRJ_TIMELINE_RESOLUTION_1280,
    timelineResolutionHeight=dcl.PRJ_TIMELINE_RESOLUTION_720,
    videoMonitorFormat=dcl.PRJ_VIDEO_MONITOR_FORMAT_HD_720P24FPS,
    timelineFrameRate=dcl.PRJ_TIMELINE_FRAMERATE_24,
    timelinePlaybackFrameRate=dcl.PRJ_TIMELINE_PLAYBACK_FRAMERATE_24,
    videoMonitorUse444SDI=dcl.PRJ_PARAM_DISABLE,
    videoMonitorSDIConfiguration=dcl.PRJ_SDI_SINGLE_LINK,
    videoDataLevels=dcl.PRJ_VIDEO_DATA_LEVEL_LIMITED,
    videoMonitorUseHDROverHDMI=dcl.PRJ_PARAM_ENABLE,
    colorScienceMode=dcl.PRJ_COLOR_SCIENCE_MODE_RCM_ON,
    isAutoColorManage=dcl.PRJ_PARAM_DISABLE,
    rcmPresetMode=dcl.PRJ_PRESET_MODE_CUSTOM,
    separateColorSpaceAndGamma=dcl.PRJ_PARAM_ENABLE,
    colorSpaceInput=dcl.PRJ_COLOR_SPACE_REC709,
    colorSpaceInputGamma=dcl.PRJ_GAMMA_STR_REC709,
    colorSpaceTimeline=dcl.PRJ_COLOR_SPACE_REC709,
    colorSpaceTimelineGamma=dcl.PRJ_GAMMA_STR_LINER,
    colorSpaceOutput=dcl.PRJ_COLOR_SPACE_REC709,
    colorSpaceOutputGamma=dcl.PRJ_GAMMA_STR_LINER,
    timelineWorkingLuminance=dcl.PRJ_WORKING_LUMINANCE_MAX,
    timelineWorkingLuminanceMode=dcl.PRJ_LUMINANCE_MODE_CUSTOM,
    inputDRT=dcl.PRJ_PARAM_NONE,
    outputDRT=dcl.PRJ_PARAM_NONE,
    hdrMasteringLuminanceMax="1000",
    hdrMasteringOn=dcl.PRJ_PARAM_ENABLE
)


def create_project(project_name="Dummy Project"):
    project_manager = dcl.get_project_manager()
    project = dcl.initialize_project(
        project_manager=project_manager, project_name=project_name)
    return project, project_manager


def create_timeline_with_settings_for_eotf(project, eotf_str: str):
    project_params = BASE_PRJ_SETTINGS.copy()
    project_params["colorSpaceInputGamma"] = eotf_str
    dcl.set_project_settings_from_dict(project=project, params=project_params)
    timeline = dcl.create_timeline(eotf_str)

    """The color settings for each timeline have been abandoned"""
    # dcl.set_timeline_settings_from_dict(
    #     timeline=timeline, params=project_params
    # )

    return timeline


def dump_data_to_log_txt(output_str):
    print(output_str)

    with open("log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(output_str + "\n")


def run_resolve_eotf(eotf_str=dcl.PRJ_GAMMA_STR_REC709_A):
    # project settings
    project_name = f"EOTF_{eotf_str}"
    # dcl.close_and_remove_project(project_name=project_name)
    # project, project_manager = create_project(project_name=project_name)
    dcl.load_project(project_name=project_name)

    dcl.open_page(dcl.EDIT_PAGE_STR)
    # create_timeline_with_settings_for_eotf(
    #     project=project, eotf_str=eotf_str
    # )

    # add clips
    media_path = str(Path('./src_img/10-bit_ramp.dpx').resolve())
    print(f"media_path = {media_path}")
    # clip_list = dcl.add_files_to_media_pool(media_path=media_path)
    # clip_property = clip_list[0].GetClipProperty()
    # print(clip_list)
    # timeline_item = dcl.add_clips_to_the_current_timeline(clip_list=clip_list)
    # media_pool_item = timeline_item[0].GetMediaPoolItem()
    # clip_property = media_pool_item.GetClipProperty()
    # input_color_space_str = "Input Color Space"
    # dump_data_to_log_txt(clip_property[input_color_space_str])
    settings = dcl.resolve.GetProjectManager().GetCurrentProject().GetSetting()
    pprint.pprint(settings)
    gamut = settings['colorSpaceTimeline']
    gamma = settings['colorSpaceTimelineGamma']
    output_str = f"{gamut}, {gamma}"
    print(output_str)

    with open("concat_input_color_space-gamut-gamma.txt", "a", encoding="utf-8") as log_file:
        log_file.write(output_str + "\n")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    eotf_str_list = [
        dcl.PRJ_GAMMA_STR_REC709_A,
    ]
    for eotf_str in eotf_str_list:
        run_resolve_eotf(eotf_str=eotf_str)
