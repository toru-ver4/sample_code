# -*- coding: utf-8 -*-
"""
color volume boundary data control
==================================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import XYZ_to_RGB, xyY_to_XYZ, RGB_COLOURSPACES

# import my libraries
from common import MeasureExecTime
import color_space as cs
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


class GamutBoundaryData():
    """
    My Gamut Boundary Data.
    This instance has a some data.

    * a pile of ab plane data
    * l values (this if for log scale data)
    """
    def __init__(self, lab):
        """
        Parameters
        ----------
        lab : ndarray
            ab value for each L and hue.
            the shape is (N, M, 3).
            N is a number of L.
            M is a number of Hue.
            "3" are L, a, and b.
        """
        self.lab = lab
        self.reduced_lab = None
        self.outline_lab = None

    def __str__(self):
        return self.lab.__str__()

    def get_as_xyY(self):
        """
        treat the internal self.lab as the xyY.
        some sorting is done.

        Returns
        -------
        ndarray
            xyY value.
        """
        xyY = self.lab.copy()
        xyY[..., 0] = self.lab[..., 1]
        xyY[..., 1] = self.lab[..., 2]
        xyY[..., 2] = self.lab[..., 0]

        return xyY

    def _calc_reduced_data(self):
        pass

    def get_reduced_data(self):
        self._calc_reduced_data()
        return self.reduced_lab

    def _calc_outline_data(self):
        pass

    def get_outline_data(self):
        self._calc_outline_data()
        return self.outline_lab


def is_inner_gamut_xyY(xyY, color_space_name=cs.BT2020, white=cs.D65):
    """
    judge the data of xyY is inner gamut.

    Parameters
    ----------
    xyY : ndarray
        xyY value.
    color_space_name : str
        name of the target color space.
    white : ndarray
        white point. ex: white=np.array([0.3127, 0.3290])

    Returns
    -------
    ndarray(bool)
        true: xyY point is inside the gamut
        false: xyY point is outside the gamut

    Examples
    --------
    >>> xyY = np.array(
    ...     [[0.3127, 0.3290, 0.5], [0.3127, 0.3290, 0.999],
    ...      [0.3127, 0.3290, 1.5], [0.60, 0.32, 0.00],
    ...      [0.60, 0.32, 0.01], [0.60, 0.32, 0.5]])
    >>> is_inner_gamut_xyY(xyY=xyY, color_space_name=cs.BT709, white=cs.D65)
    [ True  True False  True  True False]
    """
    rgb = XYZ_to_RGB(
        xyY_to_XYZ(xyY), white, white,
        RGB_COLOURSPACES[color_space_name].XYZ_to_RGB_matrix)
    r_judge = (rgb[..., 0] >= 0) & (rgb[..., 0] <= 1)
    g_judge = (rgb[..., 1] >= 0) & (rgb[..., 1] <= 1)
    b_judge = (rgb[..., 2] >= 0) & (rgb[..., 2] <= 1)
    judge = (r_judge & g_judge) & b_judge

    return judge


def calc_xyY_boundary_data_specific_Y(
        large_y=0.5, color_space_name=cs.BT2020,
        white=cs.D65, h_num=1024):
    """
    calculate the boundary data of the xyY color volume.
    this function calculates for a xy plane of specific Y.
    this function search the boundary counterclockwise.

    Parameters
    ----------
    large_y : float
        Y value. 1.0 is correspond to the white.
    color_space_name : str
        the name of color space.
    white : ndarray
        white point. ex: white=np.array([0.3127, 0.3290])
    h_num : int
        the number of samples at the counterclockwise search point.

    Returens
    --------
    ndarray
        Yxy values of the boundary data.
        the shape is (h_num, 3)

    Examples
    --------
    >>> calc_xyY_boundary_data_specific_Y(
    ...     large_y=0.01, color_space_name=cs.BT709,
    ...     white=cs.D65, h_num=16)
    [[ 0.01   0.638  0.329]
     [ 0.01   0.523  0.423]
     [ 0.01   0.45   0.481]
     [ 0.01   0.38   0.536]
     [ 0.01   0.289  0.559]
     [ 0.01   0.253  0.432]
     [ 0.01   0.239  0.382]
     [ 0.01   0.23   0.347]
     [ 0.01   0.219  0.309]
     [ 0.01   0.202  0.249]
     [ 0.01   0.161  0.066]
     [ 0.01   0.293  0.139]
     [ 0.01   0.362  0.177]
     [ 0.01   0.421  0.209]
     [ 0.01   0.493  0.249]
     [ 0.01   0.638  0.329]]
    """
    if large_y <= 0.0:
        return np.zeros((h_num, 3))

    r_val_init = 0.8
    iteration_num = 20
    hue = np.linspace(0, 2*np.pi, h_num)
    rr = np.ones(h_num) * r_val_init
    llyy = np.ones(h_num) * large_y

    for idx in range(iteration_num):
        xx = rr * np.cos(hue) + white[0]
        yy = rr * np.sin(hue) + white[1]
        xyY = np.dstack((xx, yy, llyy)).reshape((h_num, 3))
        ok_idx = is_inner_gamut_xyY(
            xyY=xyY, color_space_name=color_space_name, white=white)

        add_sub = r_val_init / (2 ** (idx + 1))
        rr[ok_idx] = rr[ok_idx] + add_sub
        rr[~ok_idx] = rr[~ok_idx] - add_sub

    xx = rr * np.cos(hue) + white[0]
    yy = rr * np.sin(hue) + white[1]

    return np.dstack((llyy, xx, yy)).reshape((h_num, 3))


def calc_xyY_boundary_data(
        color_space_name=cs.BT2020, white=cs.D65, y_num=1024, h_num=1024,
        overwirte_lut=False):
    """
    calculate the gamut boundary of xyY color volume.

    Parameters
    ----------
    color_space_name : str
        the name of the color space. ex: 'ITU-R BT.2020'.
    white : ndarray
        white point. ex: white=np.array([0.3127, 0.3290])
    y_num : int
        the number of samples for large Y.
    h_num : int
        the number of samples at the counterclockwise search point.
    overwirte_lut : bool
        whether to overwrite the LUT to recuce calculation time.

    Returns
    -------
    GamutBoundaryData
        this object includes large Y, small x and small y data.
        the shape of data is (N, M, 3).
        N is a number of large Y.
        M is a number of Hue.
        this data a stack of xy planes.

    Notes
    -----
    the first index data returned is the same as the second index data.
    the RGB primary point should be maintained even when Y=0 I think.
    please see the Examples.

    Examples
    --------
    >>> calc_xyY_boundary_data(
    ...     color_space_name=cs.BT2020, white=cs.D65, y_num=5, h_num=8,
    ...     overwirte_lut=True)
    [[[ 0.25   0.669  0.329]
      [ 0.25   0.465  0.52 ]
      [ 0.25   0.216  0.754]
      [ 0.25   0.15   0.407]
      [ 0.25   0.141  0.247]
      [ 0.25   0.269  0.139]
      [ 0.25   0.435  0.176]
      [ 0.25   0.669  0.329]]

     [[ 0.25   0.669  0.329]
      [ 0.25   0.465  0.52 ]
      [ 0.25   0.216  0.754]
      [ 0.25   0.15   0.407]
      [ 0.25   0.141  0.247]
      [ 0.25   0.269  0.139]
      [ 0.25   0.435  0.176]
      [ 0.25   0.669  0.329]]

     [[ 0.5    0.48   0.329]
      [ 0.5    0.465  0.52 ]
      [ 0.5    0.216  0.754]
      [ 0.5    0.15   0.407]
      [ 0.5    0.173  0.262]
      [ 0.5    0.289  0.226]
      [ 0.5    0.384  0.239]
      [ 0.5    0.48   0.329]]

     [[ 0.75   0.368  0.329]
      [ 0.75   0.465  0.52 ]
      [ 0.75   0.254  0.588]
      [ 0.75   0.178  0.394]
      [ 0.75   0.259  0.303]
      [ 0.75   0.303  0.286]
      [ 0.75   0.342  0.293]
      [ 0.75   0.368  0.329]]

     [[ 1.     0.313  0.329]
      [ 1.     0.313  0.329]
      [ 1.     0.313  0.329]
      [ 1.     0.313  0.329]
      [ 1.     0.313  0.329]
      [ 1.     0.313  0.329]
      [ 1.     0.313  0.329]
      [ 1.     0.313  0.329]]]
    """
    fname = f"./lut/xyY_LUT_YxH_{y_num}x{h_num}.npy"
    if os.path.isfile(fname) and (not overwirte_lut):
        largey_xy_data = np.load(fname)
        xyY_obj = GamutBoundaryData(largey_xy_data)
        return xyY_obj
    mtime = MeasureExecTime()
    mtime.start()
    out_buf = np.zeros((y_num, h_num, 3))
    y_list = np.linspace(0, 1.0, y_num)
    for idx, large_y in enumerate(y_list):
        print(f"idx = {idx} / {y_num}")
        out_buf[idx] = calc_xyY_boundary_data_specific_Y(
            large_y=large_y, color_space_name=color_space_name,
            white=white, h_num=h_num)
        mtime.lap()
    mtime.end()

    out_buf[0] = out_buf[1].copy()

    os.makedirs("./lut", exist_ok=True)
    np.save(fname, out_buf)
    return GamutBoundaryData(out_buf)


def calc_xyY_boundary_data_log_scale(
        color_space_name=cs.BT2020, white=cs.D65, y_num=1024, h_num=1024,
        min_exposure=-4, max_exposure=0, overwirte_lut=False):
    """

    Parameters
    ----------
    color_space_name : str
        the name of the color space. ex: 'ITU-R BT.2020'.
    white : ndarray
        white point. ex: white=np.array([0.3127, 0.3290])
    y_num : int
        the number of samples for large Y.
    h_num : int
        the number of samples at the counterclockwise search point.
    min_exposure : int
        minimum Y value compared to 1.0.
        ex: min_exposure=-4 --> min_Y = 10 ** (-4) = 0.0001
    max_exposure : int
        maximum Y value compared to 1.0.
        ex: max_exposure=0 --> max_Y = 10 ** (0) = 1.0
    overwirte_lut : bool
        whether to overwrite the LUT to recuce calculation time.

    Returns
    -------
    GamutBoundaryData
        this object includes large Y, small x and small y data.
        the shape of data is (N, M, 3).
        N is a number of large Y.
        M is a number of Hue.
        this data a stack of xy planes.

    Examples
    --------
    >>> calc_xyY_boundary_data_log_scale(
    ...     color_space_name=cs.BT2020, white=cs.D65, y_num=4, h_num=8,
    ...     min_exposure=-1, max_exposure=0, overwirte_lut=True)
    [[[ 0.1    0.669  0.329]
      [ 0.1    0.465  0.52 ]
      [ 0.1    0.216  0.754]
      [ 0.1    0.15   0.407]
      [ 0.1    0.141  0.247]
      [ 0.1    0.261  0.101]
      [ 0.1    0.435  0.176]
      [ 0.1    0.669  0.329]]

     [[ 0.215  0.669  0.329]
      [ 0.215  0.465  0.52 ]
      [ 0.215  0.216  0.754]
      [ 0.215  0.15   0.407]
      [ 0.215  0.141  0.247]
      [ 0.215  0.266  0.124]
      [ 0.215  0.435  0.176]
      [ 0.215  0.669  0.329]]

     [[ 0.464  0.505  0.329]
      [ 0.464  0.465  0.52 ]
      [ 0.464  0.216  0.754]
      [ 0.464  0.15   0.407]
      [ 0.464  0.157  0.254]
      [ 0.464  0.287  0.216]
      [ 0.464  0.392  0.23 ]
      [ 0.464  0.505  0.329]]

     [[ 1.     0.313  0.329]
      [ 1.     0.313  0.329]
      [ 1.     0.313  0.329]
      [ 1.     0.313  0.329]
      [ 1.     0.313  0.329]
      [ 1.     0.313  0.329]
      [ 1.     0.313  0.329]
      [ 1.     0.313  0.329]]]
    """
    fname = f"./lut/xyY_LUT_{color_space_name}_"\
        + f"exp_{min_exposure}_to_{max_exposure}_"\
        + f"Log_YxH_{y_num}x{h_num}.npy"
    y_list = tpg.get_log10_x_scale(
        sample_num=y_num, min_exposure=min_exposure,
        max_exposure=max_exposure)
    if os.path.isfile(fname) and (not overwirte_lut):
        large_y_xy = np.load(fname)
        return GamutBoundaryData(large_y_xy)
    mtime = MeasureExecTime()
    mtime.start()
    out_buf = np.zeros((y_num, h_num, 3))

    for idx, large_y in enumerate(y_list):
        print(f"idx = {idx} / {y_num}")
        out_buf[idx] = calc_xyY_boundary_data_specific_Y(
            large_y=large_y, color_space_name=color_space_name,
            white=white, h_num=h_num)
        mtime.lap()
    mtime.end()

    os.makedirs("./lut", exist_ok=True)
    np.save(fname, out_buf)
    return GamutBoundaryData(out_buf)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    xyY = np.array(
        [[0.3127, 0.3290, 0.5], [0.3127, 0.3290, 0.999],
         [0.3127, 0.3290, 1.5], [0.60, 0.32, 0.00],
         [0.60, 0.32, 0.01], [0.60, 0.32, 0.5]])
    result = is_inner_gamut_xyY(
        xyY=xyY, color_space_name=cs.BT709, white=cs.D65)

    np.set_printoptions(precision=3)
    result = calc_xyY_boundary_data_specific_Y(
        large_y=0.01, color_space_name=cs.BT709,
        white=cs.D65, h_num=16)

    result = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, white=cs.D65, y_num=5, h_num=8,
        overwirte_lut=False)
    print(result)

    result = calc_xyY_boundary_data_log_scale(
        color_space_name=cs.BT2020, white=cs.D65, y_num=4, h_num=8,
        min_exposure=-1, max_exposure=0, overwirte_lut=False)
    print(result)
