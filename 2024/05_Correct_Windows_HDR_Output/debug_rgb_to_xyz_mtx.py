# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np
from colour import sd_to_XYZ, MultiSpectralDistributions, MSDS_CMFS,\
    SDS_ILLUMINANTS, SpectralShape, XYZ_to_xy, xy_to_XYZ, SpectralDistribution
from colour.continuous import Signal
from scipy import linalg
from colour.models import eotf_ST2084, eotf_inverse_ST2084
from colour.algebra import vector_dot

# import my libraries

CIE1931_CMFS = MultiSpectralDistributions(MSDS_CMFS['cie_2_1931'])
ILLUMINANT_D65 = SDS_ILLUMINANTS['D65']

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_rgb_to_xyz_matrix(
        rgb_xy_primaries=[[[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]],
        white_point_XYZ=[0.95046, 1.00000, 1.08906]
):
    rgb_xy_primaries=[[[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]]
    rgb_xy = np.array(rgb_xy_primaries)
    rgb_z = 1 - np.sum(rgb_xy, axis=-1)
    rgb_xyz = np.concatenate((rgb_xy, rgb_z[..., np.newaxis]), axis=-1)
    p_mtx = rgb_xyz.reshape(3, 3).T
    xyz_mtx_inv = linalg.inv(p_mtx)
    t_rgb = np.dot(xyz_mtx_inv, white_point_XYZ)
    t_mtx = np.array(
        [[t_rgb[0], 0, 0], [0, t_rgb[1], 0], [0, 0, t_rgb[2]]])
    rgb_to_xyz_mtx = np.dot(p_mtx, t_mtx)

    return rgb_to_xyz_mtx


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    np.set_printoptions(precision=4)
    # spectral_shape = SpectralShape(360, 780, 5)
    # cmfs1931 = CIE1931_CMFS.copy().align(spectral_shape)
    # sds_d65 = ILLUMINANT_D65.copy().align(spectral_shape)
    # d65_5nm_xyz = sd_to_XYZ(sd=sds_d65, cmfs=cmfs1931, illuminant=None)
    # d65_5nm_xyz = d65_5nm_xyz / d65_5nm_xyz[1] * 100
    # print(d65_5nm_xyz, XYZ_to_xy(d65_5nm_xyz))

    # spectral_shape = SpectralShape(360, 780, 1)
    # cmfs1931 = CIE1931_CMFS.copy().align(spectral_shape)
    # sds_d65 = ILLUMINANT_D65.copy().align(spectral_shape)
    # d65_1nm_xyz = sd_to_XYZ(sd=sds_d65, cmfs=cmfs1931, illuminant=None)
    # d65_1nm_xyz = d65_1nm_xyz / d65_1nm_xyz[1] * 100
    # print(d65_1nm_xyz, XYZ_to_xy(d65_1nm_xyz))

    # large_xyz = np.array([95.047, 100, 108.883])
    # print(XYZ_to_xy(large_xyz))

    # spectral_shape = SpectralShape(360, 830, 1)
    # d65_csv_data = np.loadtxt("./data/d65_CIE_S_014-2.csv", delimiter=",")
    # domain = d65_csv_data[..., 0]
    # d65_raw = d65_csv_data[..., 2]
    # signal = Signal(data=d65_raw, domain=domain)
    # d65_spd = SpectralDistribution(data=signal).align(spectral_shape)
    # cmfs1931 = CIE1931_CMFS.copy().align(spectral_shape)
    # d65_1nm_xyz = sd_to_XYZ(sd=d65_spd, cmfs=cmfs1931, illuminant=None)
    # d65_1nm_xyz = d65_1nm_xyz / d65_1nm_xyz[1] * 100
    # print(d65_1nm_xyz, XYZ_to_xy(d65_1nm_xyz))

    # large_xyz = xy_to_XYZ([0.3127, 0.3290])
    # print(large_xyz)
    rec709_rgb_xy_primaries = [[[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]]
    rec2020_rgb_xy_primaries = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]]
    white_XYZ_1 = [0.95046, 1.00000, 1.08906]
    white_XYZ_2 = [0.95047, 1.00000, 1.08883]
    rec709_rgb_to_xyz_mtx_1 = calc_rgb_to_xyz_matrix(
        rgb_xy_primaries=rec709_rgb_xy_primaries, white_point_XYZ=white_XYZ_1
    )
    rec709_rgb_to_xyz_mtx_2 = calc_rgb_to_xyz_matrix(
        rgb_xy_primaries=rec709_rgb_xy_primaries, white_point_XYZ=white_XYZ_2
    )
    rec2020_rgb_to_xyz_mtx_1 = calc_rgb_to_xyz_matrix(
        rgb_xy_primaries=rec2020_rgb_xy_primaries, white_point_XYZ=white_XYZ_1
    )
    rec2020_rgb_to_xyz_mtx_2 = calc_rgb_to_xyz_matrix(
        rgb_xy_primaries=rec2020_rgb_xy_primaries, white_point_XYZ=white_XYZ_2
    )
    rec709_to_rec2020_mtx_1 = linalg.inv(rec2020_rgb_to_xyz_mtx_1)\
        .dot(rec709_rgb_to_xyz_mtx_1)
    rec2020_to_rec709_mtx_2 = linalg.inv(rec709_rgb_to_xyz_mtx_2)\
        .dot(rec2020_rgb_to_xyz_mtx_2)

    diff_rec709_mtx = rec709_rgb_to_xyz_mtx_1 - rec709_rgb_to_xyz_mtx_2
    diff_rec2020_mtx = rec2020_rgb_to_xyz_mtx_1 - rec2020_rgb_to_xyz_mtx_2
    print(diff_rec709_mtx)
    print(diff_rec2020_mtx)

    test_data_rec709_pq = [
        [1, 1, 1],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 1], [1, 1, 0], [0, 1, 1]
    ]
    test_data_rec709_linear = eotf_ST2084(np.array(test_data_rec709_pq))
    print(f"test_data_rec709_linear = {test_data_rec709_linear}")


    test_data_rec2020_linear_1\
        = vector_dot(rec709_to_rec2020_mtx_1, test_data_rec709_linear)

    test_data_rec709_linear_1_2\
        = vector_dot(rec2020_to_rec709_mtx_2, test_data_rec2020_linear_1)
    
    print(f"test_data_rec709_linear_1_2 = {test_data_rec709_linear_1_2}")
    
    test_data_rec709_pq_1_2\
        = eotf_inverse_ST2084(np.abs(test_data_rec709_linear_1_2))
    test_data_rec709_pq_1_2_10bit = np.round(test_data_rec709_pq_1_2 * 1023)
    test_data_rec709_pq_10bit = np.array(test_data_rec709_pq, dtype=np.uint16) * 1023
    diff_pq_10bit = test_data_rec709_pq_10bit - test_data_rec709_pq_10bit
    
    print(f"diff_pq_10bit = {diff_pq_10bit}")
