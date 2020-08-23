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


def add_data_to_start_and_end_for_inner_product(data):
    """
    Examples
    --------
    >>> old_data = np.array(
    ...     [[0, 0], [2, 0], [2, 2], [1, 3], [0, 2], [-1, 1]])
    >>> new_data = add_data_to_start_and_end_for_inner_product(old_data)
    >>> print(old_data)
    [[0  0] [2  0] [2  2] [1  3] [0  2] [-1  1]]
    >>> print(new_data)
    [[-1  1] [0  0] [2  0] [2  2] [1  3] [0  2] [-1  1] [ 0  0]]
    """
    temp = np.append(data[-1].reshape(1, 2), data, axis=0)
    new_data = np.append(temp, data[0].reshape(1, 2), axis=0)

    return new_data


def calc_vector_from_ndarray(data):
    """
    Examles
    -------
    >>> data = np.array(
    ...     [[0, 0], [2, 0], [2, 2], [1, 3], [0, 2], [-1, 1]])
    >>> calc_vector_from_ndarray(data)
    [[ 2  0] [ 0  2] [-1  1] [-1 -1] [-1 -1]]
    """
    return data[1:] - data[0:-1]


def calc_norm_from_ndarray(data):
    """
    Parameters
    ----------
    data : ndarray
        2-dimensional data.

    Examples
    --------
    >>> data = np.array(
    ...     [[0, 0], [2, 0], [2, 2], [1, 3], [0, 2], [-1, 1]])
    >> calc_norm_from_ndarray(data)
    [2.         2.         1.41421356 1.41421356 1.41421356]
    """
    vec = calc_vector_from_ndarray(data=data)
    norm = np.sqrt((vec[..., 0] ** 2 + vec[..., 1] ** 2))

    return norm


def calc_inner_product_from_ndarray(data):
    """
    Parameters
    ----------
    data : ndarray
        2-dimensional data.

    Examples
    --------
    >>> data = np.array(
    ...     [[0, 0], [2, 0], [2, 2], [1, 3], [0, 2], [-1, 1]])
    >> calc_inner_product(data)
    [0 2 0 2]
    """
    vec = calc_vector_from_ndarray(data=data)
    inner = vec[:-1, 0] * vec[1:, 0] + vec[:-1, 1] * vec[1:, 1]

    return inner


def calc_angle_from_ndarray(data):
    """
    Parameters
    ----------
    data : ndarray
        2-dimensional data.

    Examples
    --------
    >>> data = np.array(
    ...     [[0, 0], [2, 0], [2, 2], [1, 3], [0, 2], [-1, 1]])
    >> calc_angle_from_ndarray(data)
    [ 90.         135.          90.         179.99999879]
    """
    norm = calc_norm_from_ndarray(data=data)
    norm[norm < 10 ** -12] = 0.0
    inner = calc_inner_product_from_ndarray(data=data)

    cos_val = inner / (norm[:-1] * norm[1:])
    cos_val = np.nan_to_num(cos_val, nan=-1)

    angle = 180 - np.rad2deg(np.arccos(cos_val))

    return angle


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
        self.reduce_sample_angle_threshold = 130
        self.ab_plane_data_div_num = 40
        self.rad_rate = 4.0
        self.l_step = 1

    def __str__(self):
        return self.lab.__str__()

    def _conv_to_xyY(self, lab):
        xyY = lab.copy()
        xyY[..., 0] = lab[..., 1]
        xyY[..., 1] = lab[..., 2]
        xyY[..., 2] = lab[..., 0]

        return xyY

    def get_as_xyY(self):
        """
        treat the internal self.lab as the xyY.
        some sorting is done.

        Returns
        -------
        ndarray
            xyY value.
        """
        xyY = self._conv_to_xyY(self.lab)
        return xyY

    def _calc_reduced_data(self, threshold_angle=None):
        """
        recude the self.lab data based on the angle on ab plane.

        Returns
        -------
        array
            array of ndarray that is Lab data of edge point.
        """
        if not threshold_angle:
            threshold_angle = self.reduce_sample_angle_threshold
        out_buf = []
        for idx in range(len(self.lab)):
            ab_data = self.lab[idx, :, 1:].copy()
            ab_data = add_data_to_start_and_end_for_inner_product(ab_data)
            angle = calc_angle_from_ndarray(ab_data)
            angle[angle < 2] = 180  # remove noise data
            rd_idx = (angle < threshold_angle)
            out_buf.append(self.lab[idx, rd_idx])

        return np.vstack(out_buf)

    def get_reduced_data(self, threshold_angle=None):
        """
        Returns
        -------
        array
            array of ndarray that is Lab data of edge point.
        """
        reduced_data = self._calc_reduced_data(threshold_angle=threshold_angle)
        return reduced_data

    def get_reduced_data_as_xyY(self, threshold_angle=None):
        """
        Returns
        -------
        array
            array of ndarray that is Lab data of edge point.
        """
        reduced_data = self._calc_reduced_data(threshold_angle=threshold_angle)
        return self._conv_to_xyY(reduced_data)

    def _calc_outline_data(
            self, ab_plane_div_num=None, rad_rate=None, l_step=None):
        """
        calclurate color volume outline.

        Parameters
        ----------
        ab_plane_div_num : int
            the number mesh point on each ab plane.
        rad_rate : float
            the speed of counterclockwise sampling.
        l_step : int
            step for "L" direction.

        Returns
        -------
            array of mesh-data.
        """
        ab_plane_div_num = ab_plane_div_num\
            if ab_plane_div_num else self.ab_plane_data_div_num
        rad_rate = rad_rate if rad_rate else self.rad_rate
        l_step = l_step if l_step else self.l_step
        l_reduced_num = len(self.lab[::l_step])
        ab_plane_sample_num = len(self.lab[0])
        ab_next_idx_offset_list = tpg.equal_devision(
            int(l_reduced_num * rad_rate), l_reduced_num)
        rad_offset_list = np.linspace(0, 1, ab_plane_div_num, endpoint=False)

        out_buf = []
        for rad_st in rad_offset_list:
            ab_idx = int(rad_st * ab_plane_sample_num)
            ab_next_offset_idx = 0
            for l_idx in range(0, len(self.lab), l_step):
                lab_data = self.lab[l_idx, ab_idx % ab_plane_sample_num]
                out_buf.append(lab_data)
                ab_idx += ab_next_idx_offset_list[ab_next_offset_idx]
                ab_next_offset_idx += 1

        # inverse direction
        ab_next_idx_offset_list = tpg.equal_devision(
            int(l_reduced_num * -rad_rate), l_reduced_num)
        for rad_st in rad_offset_list:
            ab_idx = int(rad_st * ab_plane_sample_num)
            ab_next_offset_idx = 0
            for l_idx in range(0, len(self.lab), l_step):
                lab_data = self.lab[l_idx, ab_idx % ab_plane_sample_num]
                out_buf.append(lab_data)
                ab_idx += ab_next_idx_offset_list[ab_next_offset_idx]
                ab_next_offset_idx += 1
        return np.vstack(out_buf)

    def get_outline_mesh_data(
            self, ab_plane_div_num=None, rad_rate=None, l_step=None):
        """
        calclurate color volume outline.

        Parameters
        ----------
        ab_plane_div_num : int
            the number mesh point on each ab plane.
        rad_rate : float
            the speed of counterclockwise sampling.
        l_step : int
            step for "L" direction.

        Returns
        -------
            array of mesh-data.
        """
        outline_data = self._calc_outline_data(
            ab_plane_div_num=ab_plane_div_num, rad_rate=rad_rate,
            l_step=l_step)
        return outline_data

    def get_outline_mesh_data_as_xyY(
            self, ab_plane_div_num=None, rad_rate=None, l_step=None):
        outline_data = self._calc_outline_data(
            ab_plane_div_num=ab_plane_div_num, rad_rate=rad_rate,
            l_step=l_step)
        return self._conv_to_xyY(outline_data)  


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
