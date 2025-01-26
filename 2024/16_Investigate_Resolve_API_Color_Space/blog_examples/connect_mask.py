# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import os

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

project_name = "Hello World 20250125"
project_manager = resolve.GetProjectManager()
project = project_manager.GetCurrentProject()
project_manager.CloseProject(project)
project_manager.DeleteProject(project_name)
project = project_manager.CreateProject(project_name)

# add dummy video to create fusion composition
resolve.OpenPage("edit")
media_pool = project.GetMediaPool()
timeline = media_pool.CreateEmptyTimeline("Timeline1")
dummy_video_relative_path = "../videos/dummy_video_1920x1080_24P.mp4"
dummy_video_abs_path = str(Path(dummy_video_relative_path).resolve())
clip_list = resolve.GetMediaStorage().AddItemListToMediaPool([dummy_video_abs_path])
clip = clip_list[0]
clip_info = {
    "mediaPoolItem": clip,
    "recordFrame": 60*60*24+24*3,  # "01:00:03:00" -> 86472
    "startFrame": 0,
    "endFrame": 24*3,              # 24 fps * 3 sec
    "mediaType": 1                 # 1: video only, 2: autio only
}
timeline_items = media_pool.AppendToTimeline([clip_info])
timeline_item = timeline_items[0]
fusion_comp = timeline_item.AddFusionComp()

fusion_comp.Lock()

# remove MediaIn1
tool_name = "MediaIn1"
media_in = next(
    (vv for vv in fusion_comp.GetToolList().values() if vv.Name == tool_name), None
)
media_in.Delete()

# add RectangleMask node (tool)
x_pos = 3
y_pos = -1
rectangle_mask = fusion_comp.AddTool("RectangleMask", x_pos, y_pos)
rectangle_mask.SetInput("Width", 0.4)
rectangle_mask.SetInput("Haight", 0.4)

# add background node (tool)
x_pos = 3
y_pos = 1
background = fusion_comp.AddTool("Background", x_pos, y_pos)
background.SetInput("TopLeftRed", 0.75)
background.SetInput("TopLeftGreen", 0.5)
background.SetInput("TopLeftBlue", 0.25)
background.SetInput("EffectMask", rectangle_mask)

# get MediaOut1
tool_name = "MediaOut1"
media_out = next(
    (vv for vv in fusion_comp.GetToolList().values() if vv.Name == tool_name), None
)

media_out.ConnectInput("Input", background)

fusion_comp.Unlock()

resolve.OpenPage("fusion")

