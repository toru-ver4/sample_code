# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import shutil
import subprocess
import re
from pathlib import Path
from datetime import datetime

# import third-party libraries
import numpy as np

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def parse_measure_result(text):
    xyz_pattern = r"XYZ: ([\d.]+) ([\d.]+) ([\d.]+)"
    xyz_match = re.search(xyz_pattern, text)

    yxy_pattern = r"Yxy: ([\d.]+) ([\d.]+) ([\d.]+)"
    yxy_match = re.search(yxy_pattern, text)

    if not xyz_match or not yxy_match:
        raise ValueError("XYZ or Yxy values not found in the output")

    xyz_values = np.array(list(map(float, xyz_match.groups())))
    yxy_values = np.array(list(map(float, yxy_match.groups())))

    return xyz_values, yxy_values


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


def read_measure_result(csv_name="./measure_result.csv"):
    """
    Returns
    -------
    [[47.694013 50.922974 57.957539 50.922974  0.304609  0.325232]
     [47.943958 50.867195 58.210997 50.867195  0.305332  0.323949]
     [48.407519 50.638748 59.038504 50.638748  0.306212  0.320327]
     [ 1.471373  1.486711  2.34785   1.486711  0.277307  0.280198]
     [ 1.478472  1.484952  2.355368  1.484952  0.277971  0.27919 ]
     [ 1.495874  1.479372  2.389671  1.479372  0.278825  0.275749]
    """
    data = np.loadtxt(
        fname=csv_name, delimiter=',', skiprows=1, usecols=range(6))
    return data


def read_xyz(ccss_file=None):
    """
    Executes the 'spotread.exe' command to measure XYZ and Yxy values.

    This function runs the 'spotread.exe' tool to perform color measurements.
    If a CCSS file is provided, it is used for the measurement; otherwise,
    the default measurement mode is used. The function raises a
    FileNotFoundError if 'spotread.exe' is not found in the system PATH.

    Parameters
    ----------
    ccss_file : str, optional
        The path to the CCSS file to be used for the measurement.
        If None, the default measurement mode is used.

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
    PATH.
    """
    cmd = "spotread.exe"
    if shutil.which(cmd) is None:
        raise FileNotFoundError(f"{cmd} not found in system PATH.")

    if ccss_file is not None:
        args = [cmd, "-x", "-X", ccss_file, "-O"]
    else:
        args = [cmd, "-x", "-O"]

    print(" ".join(args))
    print("Measurering...", end="", flush=True)
    result = subprocess.run(args, stdout=subprocess.PIPE, text=True)
    print(" completed!!")
    large_xyz, Yxy = parse_measure_result(text=result.stdout)

    print(f"XYZ = {large_xyz}, Yxy = {Yxy}")

    return large_xyz, Yxy


def read_xyz_and_save_to_csv_file(
        result_fname="./measure_result.csv", ccss_file=None):
    """
    Measures XYZ and Yxy values using 'spotread.exe' and
    saves the results to a CSV file.

    This function first performs a color measurement using the 'read_xyz'
    function.
    If a CCSS file is specified, it is used for the measurement;
    otherwise, the default measurement mode is utilized.
    The measured values are then saved to a specified CSV file
    using 'save_measure_result'.
    The CSV file includes the large XYZ values, Yxy values,

    Parameters
    ----------
    result_fname : str, optional
        The name (or path) of the CSV file to save the measurement results.
        Default is "./measure_result.csv".
    ccss_file : str, optional
        The path to the CCSS file to be used for the measurement.
        If None, the default measurement mode is used.

    Examples
    --------
    >>> read_xyz_and_save_to_csv_file("output.csv", "example.ccss")
    # Measures using 'example.ccss', saving results to 'output.csv'.

    Notes
    -----
    Ensure that 'spotread.exe' is installed and accessible in your system's
    PATH.
    The 'save_measure_result' function is used to save the data
    in the specified CSV file format.
    """
    large_xyz, Yxy = read_xyz(ccss_file=ccss_file)

    ccss_base_name = Path(ccss_file).stem if ccss_file is not None else '-'
    save_measure_result(
        large_xyz=large_xyz, Yxy=Yxy,
        csv_name=result_fname, ccss_name=ccss_base_name)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # read_xyz()
    # read_xyz(ccss_file="./ccss/RGBLEDFamily_07Feb11.ccss")
    # read_xyz(ccss_file="./ccss/WLEDFamily_07Feb11.ccss")
    # read_measure_result()

    read_xyz_and_save_to_csv_file(
        result_fname="./measure_result/Calibration-RGBLED_no_ccss.py",
        ccss_file=None)
