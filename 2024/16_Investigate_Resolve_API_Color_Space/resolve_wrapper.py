# -*- coding: utf-8 -*-

import sys
import os
import functools
from resolve_constants import *

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
    raise ConnectionError("The DaVinci Resolve app is not running.")


class TyResolveModuleError(Exception):
    """
    Raised when a error occured inside the TyResolve module.
    """
    def __init__(self, value, message):
        self.value = value
        self.message = message
        super().__init__(f"{message}")


DEBUG_ON = True


def log_return_value(func):
    """
    デバッグモードが有効な場合のみ、関数名と戻り値をログ出力するデコレーター。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if DEBUG_ON:
            print(f"[Debug] {func.__name__}() -> {result}")
        return result
    return wrapper


def get_project_manager():
    """
    Returns
    -------
    ProjectManager (BlackmagicFusion.PyRemoteObject)
        The project manager instance

    """
    project_manager = resolve.GetProjectManager()

    return project_manager


def _close_project(project):
    """
    Parameters
    ----------
    project : Project (BlackmagicFusion.PyRemoteObject)
        A Project instance of the DaVinci Resolve

    Returns
    -------
    Bool
        Returns True if successful, and False otherwise.
    """
    project_manager = get_project_manager()
    ret_val = project_manager.CloseProject(project)

    return ret_val


@log_return_value
def close_current_project():
    """
    Returns
    -------
    Bool
        Returns True if successful, and False otherwise.    
    """
    project = get_current_project()
    ret_val = _close_project(project=project)

    return ret_val


@log_return_value
def delete_project(project_name):
    """
    Parameters
    ----------
    project_name : str
        The project name

    Returns
    -------
    Bool
        Returns True if successful, and False otherwise.
    """
    project_manager = get_project_manager()
    ret_val = project_manager.DeleteProject(project_name)

    return ret_val


@log_return_value
def create_project(project_name):
    """
    Parameters
    ----------
    project_name : str
    A project name

    Returns
    -------
    Project (BlackmagicFusion.PyRemoteObject)
        If project does not already exist
    """
    project_manager = get_project_manager()
    project = project_manager.CreateProject(project_name)

    if project is None:
        msg = f'"{project_name}" project is already exists. '
        msg += 'Please use `load_project` function instead.'
        raise TyResolveModuleError(project, msg)

    return project


@log_return_value
def save_project():
    """
    Returns
    -------
    Bool
        Returns True if successful.
    """
    project_manager = get_project_manager()
    ret_val = project_manager.SaveProject()

    if ret_val is None:
        msg = 'Failed to save project. '
        msg += 'Please save project manually.'
        raise TyResolveModuleError(project, msg)

    return ret_val


@log_return_value
def load_project(project_name):
    """
    Parameters
    ----------
    project_name : str
    A project name

    Returns
    -------
    Project (BlackmagicFusion.PyRemoteObject)
        If project does not already exist
    """
    project_manager = get_project_manager()
    project = project_manager.LoadProject(project_name)

    if project is None:
        msg = f'Failed to load "{project_name}" project. '
        msg += 'Please verify that the `project_name` is correct.'
        raise TyResolveModuleError(project, msg)

    return project


@log_return_value
def get_current_project():
    """
    Returns
    -------
    Project (BlackmagicFusion.PyRemoteObject)
        If project does not already exist
    """
    project_manager = get_project_manager()
    project = project_manager.GetCurrentProject()

    return project


@log_return_value
def get_media_pool():
    """
    Returns
    -------
    MediaPool (BlackmagicFusion.PyRemoteObject)
        A MediaPool instance
    """
    project = get_current_project()
    media_pool = project.GetMediaPool()

    return media_pool


@log_return_value
def create_empty_timeline(name="timeline_x"):
    """
    Parameters
    ----------
    name : str
        A timeline name

    Returns
    -------
    Timeline (BlackmagicFusion.PyRemoteObject)
        A empty Timeline instance
    """
    media_pool = get_media_pool()
    timeline = media_pool.CreateEmptyTimeline(name)

    if timeline is None:
        msg = f'The Timeline "{name}" already exists. '
        msg += "Please provide a different name."
        raise TyResolveModuleError(project, msg)

    return timeline


@log_return_value
def setup_project_settings(params):
    """
    Parameters
    ----------
    params : dict
        A dictionary that specify the project settings.

    Returns
    -------
    Bool
        Returns True if successful.

    Examples
    --------
    >>> params = {
    ...     "timelineResolutionWidth": "1920",
    ...     "timelineResolutionHeight": "1080",
    ...     "videoMonitorFormat": "HD 1080p 24",
    ...     "timelineFrameRate": "24",
    ...     "videoMonitorUse444SDI": "0",
    ...     "videoMonitorSDIConfiguration": "single_link",
    ...     "videoDataLevels": "Video",
    ...     "videoMonitorUseHDROverHDMI": "1",
    ...     "colorScienceMode": "davinciYRGBColorManagedv2",
    ...     "rcmPresetMode": "Custom",
    ...     "separateColorSpaceAndGamma": "1",
    ...     "colorSpaceInput": "Rec.2020",
    ...     "colorSpaceInputGamma": "ST2084",
    ...     "colorSpaceTimeline": "Rec.2020",
    ...     "colorSpaceTimelineGamma": "ST2084",
    ...     "colorSpaceOutput": "Rec.2020",
    ...     "colorSpaceOutputGamma": "ST2084",
    ...     "timelineWorkingLuminance": "10000",
    ...     "timelineWorkingLuminanceMode": "Custom",
    ...     "inputDRT": "None",
    ...     "outputDRT": "None",
    ...     "hdrMasteringLuminanceMax": "1000",
    ...     "hdrMasteringOn": "1",
    ... }
    >>> setup_project_settings(params=params)
    """
    is_success = True
    for name, value in params.items():
        result = project.SetSetting(name, value)
        if result:
            print(f'    Project.SetSetting("{name}", "{value}") -> Success')
        else:
            print(f'    Project.SetSetting("{name}", "{value}") -> Failed')
            is_success = False

    if is_success is False:
        msg = 'Project.SetSetting() was failed'
        raise TyResolveModuleError(project, msg)

    return is_success


if __name__ == '__main__':
    # sample code
    project_name = "Hello World3"

    # control the project
    close_current_project()
    delete_project(project_name=project_name)
    project = create_project(project_name=project_name)
    save_project()
    close_current_project()
    project = load_project(project_name=project_name)

    # set up the project settings

    # create timelines
    timeline = create_empty_timeline()

    # add clips

    # encode
    params = {
        "timelineResolutionWidth": "1920",
        "timelineResolutionHeight": "1080",
        "videoMonitorFormat": "HD 1080p 24",
        "timelineFrameRate": "24",
        "videoMonitorUse444SDI": "0",
        "videoMonitorSDIConfiguration": "single_link",
        "videoDataLevels": "Video",
        "videoMonitorUseHDROverHDMI": "1",
        "colorScienceMode": "davinciYRGBColorManagedv2",
        "rcmPresetMode": "Custom",
        "separateColorSpaceAndGamma": "1",
        "colorSpaceInput": "Rec.2020",
        "colorSpaceInputGamma": "ST2084",
        "colorSpaceTimeline": "Rec.2020",
        "colorSpaceTimelineGamma": "ST2084",
        "colorSpaceOutput": "Rec.2020",
        "colorSpaceOutputGamma": "ST2084",
        "timelineWorkingLuminance": "10000",
        "timelineWorkingLuminanceMode": "Custom",
        "inputDRT": "None",
        "outputDRT": "None",
        "hdrMasteringLuminanceMax": "1000",
        "hdrMasteringOn": "1",
    }
    setup_project_settings(params=params)
