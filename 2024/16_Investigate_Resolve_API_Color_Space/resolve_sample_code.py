# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
import pprint

if sys.platform == "darwin":  # macOS
    resolve_script_api = (
        "/Library/Application Support/Blackmagic Design/DaVinci Resolve/"
        "Developer/Scripting"
    )
    sys.path.append(os.path.join(resolve_script_api, "Modules"))
elif sys.platform == "win32":  # Windows
    resolve_script_api = os.path.expandvars(
        r"%PROGRAMDATA%\Blackmagic Design\DaVinci Resolve\Support"
        r"\Developer\Scripting"
    )
    sys.path.append(os.path.join(resolve_script_api, "Modules"))
elif sys.platform == "linux":  # Linux
    resolve_script_api = "/opt/resolve/Developer/Scripting"
    sys.path.append(os.path.join(resolve_script_api, "Modules"))

import DaVinciResolveScript as dvr_script


resolve = dvr_script.scriptapp("Resolve")
if resolve is None:
    raise ConnectionError("The external application is not running.")

project_manager = resolve.GetProjectManager()
project_manager.CloseProject(project_manager.GetCurrentProject())
project_manager.DeleteProject("Hellow, World4")
project = project_manager.CreateProject("Hellow, World4")

# Project Settings
project.SetSetting("timelineResolutionWidth", "1920")
project.SetSetting("timelineResolutionHeight", "1080")
project.SetSetting("videoMonitorFormat", "HD 1080p 24")
project.SetSetting("timelineFrameRate", "24")
project.SetSetting("videoMonitorUse444SDI", "0")
project.SetSetting("videoMonitorSDIConfiguration", "single_link")
project.SetSetting("videoDataLevels", "Video")
project.SetSetting("videoMonitorUseHDROverHDMI", "1")
project.SetSetting("colorScienceMode", "davinciYRGBColorManagedv2")
project.SetSetting("rcmPresetMode", "Custom")
project.SetSetting("separateColorSpaceAndGamma", "1")
project.SetSetting("colorSpaceInput", "Rec.2020")
project.SetSetting("colorSpaceInputGamma", "ST2084")
project.SetSetting("colorSpaceTimeline", "Rec.2020")
project.SetSetting("colorSpaceTimelineGamma", "ST2084")
project.SetSetting("colorSpaceOutput", "Rec.2020")
project.SetSetting("colorSpaceOutputGamma", "ST2084")
project.SetSetting("timelineWorkingLuminance", "10000")
project.SetSetting("timelineWorkingLuminanceMode", "Custom")
project.SetSetting("inputDRT", "None")
project.SetSetting("outputDRT", "None")
project.SetSetting("hdrMasteringLuminanceMax", "1000")
project.SetSetting("hdrMasteringOn", "1")

resolve.OpenPage("media")
media_storage = resolve.GetMediaStorage()
media_path = str((Path(__file__).parent / "clip").resolve())
clip_list = media_storage.AddItemListToMediaPool(media_path)

clip_properties = clip_list[0].GetClipProperty()
pprint.pprint(clip_properties)

ret_value = clip_list[0].SetClipProperty('Input Color Space', 'P3-D65')
print(f"ret_value = {ret_value}")


    # dcl.open_page(dcl.EDIT_PAGE_STR)
    # create_timeline_with_settings_for_oetf(
    #     project=project, oetf_str=oetf_str
    # )

    # # add clips
    # media_path = str(Path('./src_img/src_log2_-12.551_to_2.474_stops.exr').resolve())
    # print(f"media_path = {media_path}")
    # clip_list = dcl.add_files_to_media_pool(media_path=media_path)
