# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import re
import sys
import shutil
import importlib
import subprocess
from pathlib import Path
from datetime import datetime

# import third-party libraries

# import my libraries
try:
    import ty_davinci_control_lib as dcl
except ImportError:
    try:
        sys.path.append('C:/Users/toruv/OneDrive/work/sample_code/ty_lib')
        print("ty_davinci_control_lib is not found - trying another locations")
        import ty_davinci_control_lib as dcl
    except ImportError:
        msg = "Failed to import 'ty_davinci_control_lib'. "
        msg += "Please check the installation and PYTHONPATH."
        print(msg)
importlib.reload(dcl)

"""
import sys
import importlib
folder_path = 'C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve'
sys.path.append(folder_path)
import measure_with_resolve as mwr
importlib.reload(mwr)
mwr.simple_measure()
"""


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


CCSS_RGBLED = "C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve/ccss/RGBLEDFamily_07Feb11.ccss"


def save_measure_result(
        large_xyz, Yxy, csv_name="./measure_result.csv", ccss_name='-'):
    """
    Saves the measurement results in a CSV file, including the current time.

    This function writes the provided XYZ and Yxy measurement values
    to a specified CSV file. 
    If the file does not exist, it is created with a header.
    If the file exists, the data is appended.
    Each entry is timestamped with the current time of the measurement.

    Parameters
    ----------
    large_xyz : tuple
        A tuple containing the XYZ measurement values.
    Yxy : tuple
        A tuple containing the Yxy measurement values.
    csv_name : str, optional
        The name (or path) of the CSV file to save the results to.
        Default is "./measure_result.csv".
    ccss_name : str, optional
        The name of the CCSS file used, if any. Default is '-'.

    Examples
    --------
    >>> save_measure_result(
    ...     (47.694013, 50.922974, 57.957539), (50.922974, 0.304609, 0.325232))
    # This will write the provided XYZ and Yxy values to 'measure_result.csv'.

    Notes
    -----
    A sample.csv is below

    ```
    X,Y,Z,Y,x,y,ccss_fname,Date
    47.694013,50.922974,57.957539,50.922974,0.304609,0.325232,-,2024-01-21T23:11:05.026035
    47.943958,50.867195,58.210997,50.867195,0.305332,0.323949,WLEDFamily_07Feb11,2024-01-21T23:11:06.747487
    48.407519,50.638748,59.038504,50.638748,0.306212,0.320327,OLEDFamily_20Jul12,2024-01-21T23:11:08.474263
    1.471373,1.486711,2.34785,1.486711,0.277307,0.280198,-,2024-01-21T23:11:41.198455
    1.478472,1.484952,2.355368,1.484952,0.277971,0.27919,WLEDFamily_07Feb11,2024-01-21T23:11:48.597962
    1.495874,1.479372,2.389671,1.479372,0.278825,0.275749,OLEDFamily_20Jul12,2024-01-21T23:11:56.008590
    ```
    """
    file_path = Path(csv_name)
    file_exists = file_path.is_file()
    current_time = datetime.now().isoformat()

    with open(file_path, 'a' if file_exists else 'w') as file:
        if not file_exists:
            # add header if it's new file
            file.write("X,Y,Z,Y,x,y,ccss_fname,Date\n")

        file.write(f"{large_xyz[0]},{large_xyz[1]},{large_xyz[2]},")
        file.write(f"{Yxy[0]},{Yxy[1]},{Yxy[2]},")
        file.write(f"{ccss_name},{current_time}\n")


def parse_measure_result(text):
    xyz_pattern = r"XYZ: ([\d.]+) ([\d.]+) ([\d.]+)"
    xyz_match = re.search(xyz_pattern, text)

    yxy_pattern = r"Yxy: ([\d.]+) ([\d.]+) ([\d.]+)"
    yxy_match = re.search(yxy_pattern, text)

    if not xyz_match or not yxy_match:
        raise ValueError("XYZ or Yxy values not found in the output")

    # xyz_values = np.array(list(map(float, xyz_match.groups())))
    # yxy_values = np.array(list(map(float, yxy_match.groups())))
    xyz_values = list(map(float, xyz_match.groups()))
    yxy_values = list(map(float, yxy_match.groups()))

    return xyz_values, yxy_values


def read_xyz(ccss_file=None, flush=True):
    """
    Executes the 'spotread.exe' command to measure XYZ and Yxy values.

    This function runs the 'spotread.exe' tool to perform color measurements.
    If a CCSS file is provided, it is used for the measurement; otherwise,
    the default measurement mode is used. The function raises a
    FileNotFoundError if 'spotread.exe' is not found in the system PATH.
    The print statements within this function can be immediately flushed to the
    output, which is controlled by the 'flush' parameter. This is particularly
    useful for environments that require immediate feedback to the user.

    Parameters
    ----------
    ccss_file : str, optional
        The path to the CCSS file to be used for the measurement.
        If None, the default measurement mode is used.
    flush : bool, default True
        Controls whether to flush the output immediately. Set to False
        when calling from environments like DaVinci Resolve, where immediate
        flushing is not desired.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two numpy arrays. The first array represents 
        the large XYZ values, and the second array represents the Yxy values.

    Examples
    --------
    >>> large_xyz, Yxy = read_xyz("example.ccss")
    >>> print(large_xyz)
    [47.705729 50.903721 57.911968]
    >>> print(Yxy)
    [50.903721  0.304787  0.325219]

    Notes
    -----
    Ensure that 'spotread.exe' is installed and accessible in your system's
    PATH. The 'flush' parameter is particularly useful for scripts running
    in interactive shells or within environments that buffer output.
    """
    cmd = "spotread.exe"
    if shutil.which(cmd) is None:
        raise FileNotFoundError(f"{cmd} not found in system PATH.")

    if ccss_file is not None:
        args = [cmd, "-x", "-X", ccss_file, "-O"]
    else:
        args = [cmd, "-x", "-O"]

    print(" ".join(args))
    print("Measurering...", end="", flush=flush)
    result = subprocess.run(args, stdout=subprocess.PIPE, text=True)
    print(" completed!!")
    large_xyz, Yxy = parse_measure_result(text=result.stdout)

    print(f"XYZ = {large_xyz}, Yxy = {Yxy}")

    return large_xyz, Yxy


def create_project(project_name="Dummy Project"):
    project_manager = dcl.init_resolve()
    project = dcl.initialize_project(
        project_manager=project_manager, project_name=project_name)
    return project


def remove_all_timeline(project):
    num_of_timeline = project.GetTimelineCount()
    print(f"num_of_timeline = {num_of_timeline}")
    media_pool = project.GetMediaPool()

    timelines = []

    for idx in range(num_of_timeline):
        timeline = project.GetTimelineByIndex(idx + 1)
        timelines.append(timeline)

    if timelines != []:
        media_pool.DeleteTimelines(timelines)


def set_project_settings_bt2100(project):
    project_params = dict(
        timelineResolutionHeight="1080",
        timelineResolutionWidth="1920",
        videoMonitorFormat="HD 1080p 24",
        timelineFrameRate="24.000",
        timelinePlaybackFrameRate="24",
        videoDataLevels="Video",
        videoMonitorUseHDROverHDMI="1",
        colorScienceMode="davinciYRGBColorManagedv2",
        rcmPresetMode="Custom",
        separateColorSpaceAndGamma="1",
        colorSpaceInput="Rec.2020",
        colorSpaceInputGamma="Gamma 2.4",
        colorSpaceTimeline="Rec.2020",
        colorSpaceTimelineGamma="Gamma 2.4",
        colorSpaceOutput="Rec.2020",
        colorSpaceOutputGamma="Gamma 2.4",
        inputDRT="None",
        outputDRT="None",
    )
    dcl.set_project_settings_from_dict(project=project, params=project_params)


def simple_measure(
        csv_name="./measure_result.csv", ccss_file=None):
    # open dummy project for debug
    create_project(project_name="Dummy Project")

    project_name = "Measure_AW3225QF"
    project = create_project(project_name=project_name)
    dcl.open_page(dcl.EDIT_PAGE_STR)
    remove_all_timeline(project=project)

    # large_xyz, Yxy = read_xyz(flush=False, ccss_file=ccss_file)
    # ccss_name = Path(ccss_file).stem if ccss_file else "-"
    # save_measure_result(
    #     large_xyz=large_xyz, Yxy=Yxy,
    #     csv_name=csv_name, ccss_name=ccss_name)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    simple_measure(csv_name="./measure_result.csv", ccss_file=CCSS_RGBLED)
