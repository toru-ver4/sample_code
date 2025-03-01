# -*- coding: utf-8 -*-
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

project_name = "Apply Spline Editor Result 20250215"
project_manager = resolve.GetProjectManager()
project = project_manager.GetCurrentProject()
project_manager.CloseProject(project)
project_manager.DeleteProject(project_name)
project = project_manager.CreateProject(project_name)

# add fusion composition
resolve.OpenPage("edit")
media_pool = project.GetMediaPool()
timeline = media_pool.CreateEmptyTimeline("Timeline1")
timeline_item = timeline.InsertFusionCompositionIntoTimeline()
fusion_comp = timeline_item.AddFusionComp()

fusion_comp.Lock()

# add background node (tool)
x_pos = 1
y_pos = 1
base_bg = fusion_comp.AddTool("Background", x_pos, y_pos)
base_bg.SetInput("TopLeftRed", 0.5)
base_bg.SetInput("TopLeftGreen", 0.5)
base_bg.SetInput("TopLeftBlue", 0.5)

# add merge node (tool)
x_pos = 2
y_pos = 1
merge = fusion_comp.AddTool("Merge", x_pos, y_pos)

# add rectangle mask (tool)
x_pos = 2
y_pos = 3
rect_mask = fusion_comp.AddTool("RectangleMask", x_pos, y_pos)
rect_mask.SetInput("Width", 0.3)
rect_mask.SetInput("Height", 0.3)

# rectangle background (tool)
x_pos = 2
y_pos = 2
rect_bg = fusion_comp.AddTool("Background", x_pos, y_pos)
rect_bg.SetInput("TopLeftRed", 0.6666666)
rect_bg.SetInput("TopLeftGreen", 1.0)
rect_bg.SetInput("TopLeftBlue", 0.0)
rect_bg.SetInput("EffectMask", rect_mask)

# get MediaOut1
tool_name = "MediaOut1"
media_out = next(
    (vv for vv in fusion_comp.GetToolList().values() if vv.Name == tool_name), None
)

# connect tools
merge.ConnectInput("Background", base_bg)
merge.ConnectInput("Foreground", rect_bg)
media_out.ConnectInput("Input", merge)

key_frames = {
    0: {
        1: 0.0, 'LH': {1: 0.0, 2: 0.0}, 'RH': {1: 20.0, 2: 0.0}
    },
    23: {
        1: 1.0, 'LH': {1: -20.0, 2: 0.0}, 'RH': {1: 0.0, 2: 0.0}
    },
}
bezier_spline = fusion_comp.BezierSpline()
bezier_spline.SetKeyFrames(key_frames)
rect_mask["Width"] = bezier_spline

fusion_comp.Unlock()

resolve.OpenPage("fusion")
