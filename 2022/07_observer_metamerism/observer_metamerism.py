# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os
from pathlib import Path
import requests

# import third-party libraries
import numpy as np
from colour.continuous import MultiSignals
from colour import sd_to_XYZ, MultiSpectralDistributions, MSDS_CMFS,\
    SDS_ILLUMINANTS, SpectralShape
from colour import XYZ_to_xyY
from colour.plotting import plot_multi_sds

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def download_file(url, color_checker_sr_fname):
    response = requests.get(url)
    open(color_checker_sr_fname, "wb").write(response.content)


def prepaere_color_checker_sr_data():
    color_checker_sr_fname = "./ref_data/color_checker_sr.txt"
    url = "https://babelcolor.com/index_htm_files/CC_Avg30_spectrum_CGATS.txt"

    data = np.loadtxt(
        fname=color_checker_sr_fname, delimiter='\t', skiprows=1).T
    print(data)
    domain = np.arange(380, 740, 10)
    spectral_shape = SpectralShape(380, 730, 1)
    color_checker_signals = MultiSignals(data=data, domain=domain)
    color_checker_sds = MultiSpectralDistributions(data=color_checker_signals)
    color_checker_sds = color_checker_sds.interpolate(shape=spectral_shape)
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    cmfs = cmfs.trim(spectral_shape)
    illuminant = SDS_ILLUMINANTS['D65']
    illuminant = illuminant.interpolate(spectral_shape)
    # print(illuminant)
    XYZ = sd_to_XYZ(sd=color_checker_sds, cmfs=cmfs, illuminant=illuminant)
    xyY = XYZ_to_xyY(XYZ)
    print(xyY)


def load_color_checker_sr():
    prepaere_color_checker_sr_data()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_color_checker_sr()
