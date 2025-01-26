# -*- coding: utf-8 -*-
import sys
import os
import itertools

if sys.platform == "darwin":  # macOS
    resolve_script_api = (
        "/Library/Application Support/Blackmagic Design/DaVinci Resolve/"
        "Developer/Scripting"
    )
    sys.path.append(os.path.join(resolve_script_api, "Modules"))
    resolve_lut_path = \
        "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT/"
elif sys.platform == "win32":  # Windows
    resolve_script_api = os.path.expandvars(
        r"%PROGRAMDATA%\Blackmagic Design\DaVinci Resolve\Support"
        r"\Developer\Scripting"
    )
    sys.path.append(os.path.join(resolve_script_api, "Modules"))
    resolve_lut_path = os.path.expandvars(
        r"%PROGRAMDATA%\Blackmagic Design\DaVinci Resolve\Support\LUT"
    )
elif sys.platform == "linux":  # Linux
    resolve_script_api = "/opt/resolve/Developer/Scripting"
    sys.path.append(os.path.join(resolve_script_api, "Modules"))
    resolve_lut_path = "/home/resolve/LUT"
    ValueError("Linux platform is not supported")

import DaVinciResolveScript as dvr_script

resolve = dvr_script.scriptapp("Resolve")
if resolve is None:
    raise ConnectionError("The DaVinci Resolve app is not running.")

# get fusion composition of the current project
project = resolve.GetProjectManager().GetCurrentProject()
timeline = project.GetCurrentTimeline()
track_type = "video"
track_idx = 1
timeline_item_list = timeline.GetItemListInTrack(track_type, track_idx)
timeline_item = timeline_item_list[0]
fusion_comp = timeline_item.GetFusionCompByIndex(1)

# get Merge1 and MediaOut1 node (tool)
tool_name = "Merge1"
merge = next(
    (tool for tool in fusion_comp.GetToolList().values() if tool.Name == tool_name),
    None
)
tool_name = "MediaOut1"
media_out = next(
    (tool for tool in fusion_comp.GetToolList().values() if tool.Name == tool_name),
    None
)

# dump Merge1 MainInputs
separator = "=" * 80
print(separator)
print(f" {merge.Name} MainInput List")
print(separator)
for idx in itertools.count(1):
    main_input = merge.FindMainInput(idx)
    if main_input is None:
        break
    print(f"{idx}: Name = {main_input.Name}, ID = {main_input.ID}")

# dump MediaOut1 MainInputs
print("")
separator = "=" * 80
print(separator)
print(f" {media_out.Name} MainInput List")
print(separator)
for idx in itertools.count(1):
    main_input = media_out.FindMainInput(idx)
    if main_input is None:
        break
    print(f"{idx}: Name = {main_input.Name}, ID = {main_input.ID}")