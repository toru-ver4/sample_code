# -*- coding: utf-8 -*-

import sys
import os

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


def get_resolve():
    """
    Returns
    -------
        BlackmagicFusion.PyRemoteObject
            The Resolve instance
    """
    resolve = dvr_script.scriptapp("Resolve")

    if resolve is None:
        raise ConnectionError("The external application is not running.")

    return resolve


def get_project_manager():
    """
    Returns
    -------
        ProjectManager (BlackmagicFusion.PyRemoteObject)
            The project manager instance

    """
    resolve = get_resolve()
    project_manager = resolve.GetProjectManager()

    return project_manager


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
        None
            If project is already exists.
    """
    project_manager = get_project_manager()
    ret_value = project_manager.CreateProject(project_name)

    return ret_value


if __name__ == '__main__':
    project_name = "Hello World3"
    create_project(project_name=project_name)
    # projectManager.LoadProject()
