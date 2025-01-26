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

# Set up the project settings
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

# Add clips
resolve.OpenPage("media")
media_storage = resolve.GetMediaStorage()
media_path = str((Path(__file__).parent / "clip").resolve())
media_pool_item_list = media_storage.AddItemListToMediaPool(media_path)
clip = media_pool_item_list[0]

# Chage the clip property
color_gamut_list = [
    'Sony S-Gamut3', 'Canon Cinema Gamut', 'REDWideGamutRGB'
]
color_space_list = [
    'S-Gamut3/S-Log3', 'Canon Cinema Gamut/Canon Log 2',
    'REDWideGamutRGB/Log3G10'
]
gamma_list = [
    'S-Log3', 'Canon Log 2', 'RED Log3G10'
]

# 'Input Color Space' のパラメータを Color Gamut として設定すると成功する
print("Change the clip's Input Color Space (as the color gamut)")
for color_gamut in color_gamut_list:
    ret_value = clip.SetClipProperty('Input Color Space', color_gamut)
    debug_str = f"  clip.SetClipProperty('Input Color Space', '{color_gamut}')"
    debug_str += f" -> {ret_value}"
    print(debug_str)

# 'Input Color Space' のパラメータを Color Space として設定すると成功する
print("Change the clip's Input Color Space (as the color space)")
for color_space in color_space_list:
    ret_value = clip.SetClipProperty('Input Color Space', color_space)
    debug_str = f"  clip.SetClipProperty('Input Color Space', '{color_space}')"
    debug_str += f" -> {ret_value}"
    print(debug_str)

# 'Input Gamma' はそもそも propertyName に存在していない
print("Change the clip's Input Gamma")
for gamma in gamma_list:
    ret_value = clip.SetClipProperty('Input Gamma', gamma)
    debug_str = f"  clip.SetClipProperty('Input Gamma', '{gamma}')"
    debug_str += f" -> {ret_value}"
    print(debug_str)

# 参考情報として MediaPoolItem.SetClipProperty で設定可能なパラメータを出力
clip_properties = clip.GetClipProperty()
pprint.pprint(clip_properties)
