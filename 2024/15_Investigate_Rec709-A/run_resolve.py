# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

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


def create_timeline_with_settings(project, eotf_str: str):
    project_params = BASE_PRJ_SETTINGS.copy()
    project_params["colorSpaceInputGamma"] = eotf_str
    dcl.set_project_settings_from_dict(project=project, params=project_params)
    timeline = dcl.create_timeline(eotf_str)
    dcl.set_timeline_settings_from_dict(
        timeline=timeline, params=project_params
    )


def run_resolve_eotf(eotf_str=dcl.PRJ_GAMMA_STR_REC709_A):
    project_name = "EOTF_Investigation"
    dcl.close_and_remove_project(project_name=project_name)
    project, project_manager = create_project(project_name=project_name)
    dcl.open_page(dcl.EDIT_PAGE_STR)
    dcl.remove_all_timeline(project=project)
    create_timeline_with_settings(project=project, eotf_str=eotf_str)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_resolve_eotf(eotf_str=dcl.PRJ_GAMMA_STR_REC709)
