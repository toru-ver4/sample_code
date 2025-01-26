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
project_manager.CreateProject(project_name)

