# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os
from cv2 import illuminationChange

# import third-party libraries
import numpy as np
from scipy.stats import norm
from colour import MultiSpectralDistributions, SpectralShape, MSDS_CMFS,\
    SDS_ILLUMINANTS, sd_to_XYZ, xy_to_XYZ, XYZ_to_xyY
from colour.continuous import MultiSignals
from colour.utilities import tstack
from scipy import linalg

# import my libraries
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


CIE1931_CMFS = MultiSpectralDistributions(MSDS_CMFS["cie_2_1931"])
CIE2012_CMFS = MultiSpectralDistributions(
    MSDS_CMFS["CIE 2012 2 Degree Standard Observer"])

ILLUMINANT_E = SDS_ILLUMINANTS['E']
START_WAVELENGTH = 360
STOP_WAVELENGTH = 780
WAVELENGTH_STEP = 1


def calc_primaries_and_white(spd):
    """
    Parameters
    ----------
    msd : MultiSpectralDistributions
        A spectral distributions of the display.

    Examples
    --------
    >>> primaries, white = calc_primaries_and_white(spd)
    >>> print(primaries)
    [[  6.09323326e-01   3.87358311e-01   7.51463229e+00]
     [  3.49630616e-01   6.07805644e-01   2.97221267e+01]
     [  1.54628080e-01   2.02731412e-02   9.69632903e-01]
     [  6.09323326e-01   3.87358311e-01   7.51463229e+00]]
    >>> print(white)
    [  0.3127       0.329       38.20639185]
    """
    spd, cmfs, illuminant = trim_and_interpolate_in_advance(
        spd=spd, cmfs=CIE1931_CMFS, illuminant=ILLUMINANT_E,
        spectral_shape=SpectralShape(360, 780, 1))
    rgbw_large_xyz = sd_to_XYZ(sd=spd, cmfs=cmfs, illuminant=illuminant)
    rgbw_xyY = XYZ_to_xyY(rgbw_large_xyz)
    # print(rgbw_large_xyz)
    primaries = rgbw_xyY[:3]
    primaries = np.append(primaries, [primaries[0, :]], axis=0)
    white = rgbw_xyY[3]

    return primaries, white


def trim_and_interpolate_in_advance(
        spd, cmfs, illuminant,
        spectral_shape=SpectralShape(360, 780, 1)):
    spd2 = spd.interpolate(shape=spectral_shape)
    cmfs2 = cmfs.trim(spectral_shape)
    illuminant2 = illuminant.interpolate(shape=spectral_shape)

    return spd2, cmfs2, illuminant2


def add_white_spectrum(spd: MultiSpectralDistributions):
    # add white data to spd
    r, g, b = [spd.values[..., idx] for idx in range(3)]
    w = r + g + b
    signals_with_white = MultiSignals(
        data=tstack([r, g, b, w]), domain=spd.domain)
    spd_with_white = MultiSpectralDistributions(data=signals_with_white)

    return spd_with_white


def calc_rgb_to_xyz_matrix_from_spectral_distribution(
        spd: MultiSpectralDistributions):
    """
    Calculate RGB to XYZ matrix from spectral distribution of the display.

    Parameters
    ----------
    spd : MultiSpectralDistributions
        Spectral distribution of the display.
        Shape is `SpectralShape(360, 780, 1)`
    """
    spectral_shape = SpectralShape(
        START_WAVELENGTH, STOP_WAVELENGTH, WAVELENGTH_STEP)
    cmfs = CIE1931_CMFS
    illuminant = ILLUMINANT_E

    # spd = add_white_spectrum(spd=spd)
    spd, cmfs, illuminant = trim_and_interpolate_in_advance(
        spd=spd, cmfs=cmfs, illuminant=illuminant,
        spectral_shape=spectral_shape)
    rgbw_large_xyz = sd_to_XYZ(
        sd=spd, cmfs=cmfs, illuminant=illuminant)

    # calc RGB to XYZ matrix
    rgbw_large_xyz_sum = np.sum(rgbw_large_xyz, -1).reshape(4, 1)
    rgbw_small_xyz = (rgbw_large_xyz / rgbw_large_xyz_sum)

    xyz_mtx = rgbw_small_xyz[:3].T
    xyz_mtx_inv = linalg.inv(xyz_mtx)

    w_large_xyz = rgbw_large_xyz[3]
    w_large_xyz = w_large_xyz / w_large_xyz[1]

    t_rgb = np.dot(xyz_mtx_inv, w_large_xyz)

    t_mtx = np.array(
        [[t_rgb[0], 0, 0], [0, t_rgb[1], 0], [0, 0, t_rgb[2]]])
    rgb_to_xyz_mtx = np.dot(xyz_mtx, t_mtx)

    return rgb_to_xyz_mtx


def calc_xyz_to_rgb_matrix_from_spectral_distribution(spd):
    """
    Calculate XYZ to RGB matrix from spectral distribution of the display.

    Parameters
    ----------
    spd : MultiSpectralDistributions
        spectral distribution of the display.
        shape is `SpectralShape(360, 780, 1)`
    """
    rgb_to_xyz_mtx = calc_rgb_to_xyz_matrix_from_spectral_distribution(spd)
    return linalg.inv(rgb_to_xyz_mtx)


def calculate_white_balance_gain(msd, target_white=cs.D65):
    """
    msd : MultiSpectralDistributions
        multi spectral distributions.
    target_white : array_like
        xy coordinate of the white point.
    """
    xyz_to_rgb_mtx = calc_xyz_to_rgb_matrix_from_spectral_distribution(spd=msd)
    white_large_xyz = xy_to_XYZ(target_white)

    white_balance_gain = np.dot(xyz_to_rgb_mtx, white_large_xyz)

    return white_balance_gain


def create_display_sd(
        r_mu, r_sigma, g_mu, g_sigma, b_mu, b_sigma,
        target_white=[0.3127, 0.3290], normalize_y=False):
    """
    Create display spectral distributions using normal distribution.
    """
    st_wl = 380
    ed_wl = 780
    wl_step = 1
    x = np.arange(st_wl, ed_wl+wl_step, wl_step)

    rr = norm.pdf(x, loc=r_mu, scale=r_sigma)
    gg = norm.pdf(x, loc=g_mu, scale=g_sigma)
    bb = norm.pdf(x, loc=b_mu, scale=b_sigma)
    ww = rr + gg + bb

    # calculate rgb_gain to adjust D65
    signals = MultiSignals(data=tstack([rr, gg, bb, ww]), domain=x)
    msd = MultiSpectralDistributions(data=signals)
    gain = calculate_white_balance_gain(msd=msd, target_white=target_white)

    # apply rgb_gain and re-generate spectral distributions
    rr = rr * gain[0]
    gg = gg * gain[1]
    bb = bb * gain[2]
    ww = rr + gg + bb

    gained_sd = tstack([rr, gg, bb, ww])
    gained_sd = gained_sd / np.max(gained_sd)
    signals = MultiSignals(data=gained_sd, domain=x)
    msd = MultiSpectralDistributions(data=signals)

    # normalize to Y=100
    if normalize_y:
        _, white_xyY = calc_primaries_and_white(spd=msd)
        gained_sd = gained_sd / white_xyY[2] * 100
        signals = MultiSignals(data=gained_sd, domain=x)
        msd = MultiSpectralDistributions(data=signals)

    return msd

    # # debug plot
    # import plot_utility as pu
    # fig, ax1 = pu.plot_1_graph()
    # ax1.plot(x, rr, '--', color=pu.RED)
    # ax1.plot(x, gg, '--', color=pu.GREEN)
    # ax1.plot(x, bb, '--', color=pu.BLUE)

    # ax1.plot(x, gained_sd[..., 0], '-', color=pu.RED)
    # ax1.plot(x, gained_sd[..., 1], '-', color=pu.GREEN)
    # ax1.plot(x, gained_sd[..., 2], '-', color=pu.BLUE)
    # ax1.plot(x, gained_sd[..., 3], '-', color='k')
    # pu.show_and_save(
    #     fig=fig, save_fname="./figure/normal_distribution_test.png")


class DisplaySpectrum():
    """
    Parameters
    ----------
    msd : MultiSpectralDistributions
        A spectral distributions of the display.

    Attributes
    ----------
    cmf : MultiSpectralDistributions
    illuminant : SpectralDistribution
    primaries : ndarray

    Methods
    -------
    interpolate_and_trim
        interpolate and trim each spectral distributions
    update_msd
        update `msd`
    calc_rgbw_chromaticity
        calculate xy coordinates of the r, g, b, w.
    create_msd_based_on_metamerism
        output `msd` based on input XYZ value.

    Notes
    -----
    If you want to plot primaries, you can extend data using
    `plot_utility.add_first_value_to_end`.

    """
    def __init__(
            self, msd: MultiSpectralDistributions) -> None:
        self.spectral_shape = SpectralShape(
            START_WAVELENGTH, STOP_WAVELENGTH, WAVELENGTH_STEP)
        self.msd, self.cmfs, self.illuminant =\
            trim_and_interpolate_in_advance(
                spd=msd, cmfs=CIE1931_CMFS, illuminant=ILLUMINANT_E,
                spectral_shape=self.spectral_shape)
        self.primaries, self.white = self._calc_rgbw_chromaticity()

    def _calc_rgbw_chromaticity(self):
        rgbw_large_xyz = sd_to_XYZ(
            sd=self.msd, cmfs=self.cmfs, illuminant=self.illuminant)
        rgbw_xyY = XYZ_to_xyY(rgbw_large_xyz)
        primaries = rgbw_xyY[:3]
        primaries = np.append(primaries, [primaries[0, :]], axis=0)
        white = rgbw_xyY[3]

        return primaries, white

    def update_msd(self, msd: MultiSpectralDistributions, reshape=False):
        if reshape:
            self.msd = msd.interpolate(shape=self.spectral_shape)
        else:
            self.msd = msd

        self.primaries, self.white = self._calc_rgbw_chromaticity()

    def get_rgb_to_xyz_mtx(self):
        return calc_xyz_to_rgb_matrix_from_spectral_distribution(
            spd=self.msd)

    def calc_msd_from_rgb_gain(self, rgb: np.ndarray):
        """
        calculate new msd based on input rgb gain.

        Parameters
        ----------
        rgb : ndarray
            It's shape must be (N, 3).

        Returns
        -------
        MultiSpectralDistributions
            spectral distribution of the display applied rgb gain.
        """
        r_sd = np.reshape(self.msd.values[..., 0], (-1, 1))
        g_sd = np.reshape(self.msd.values[..., 1], (-1, 1))
        b_sd = np.reshape(self.msd.values[..., 2], (-1, 1))

        r_gain = np.reshape(rgb[..., 0], (1, -1))
        g_gain = np.reshape(rgb[..., 1], (1, -1))
        b_gain = np.reshape(rgb[..., 2], (1, -1))

        r_sd_new = r_sd * r_gain
        g_sd_new = g_sd * g_gain
        b_sd_new = b_sd * b_gain

        gained_sd = r_sd_new + g_sd_new + b_sd_new

        domain = self.msd.domain
        gained_signal = MultiSignals(data=gained_sd, domain=domain)
        gained_msd = MultiSpectralDistributions(data=gained_signal)

        return gained_msd


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    msd = create_display_sd(
        r_mu=620, r_sigma=12, g_mu=535, g_sigma=18, b_mu=458, b_sigma=8)
    # primaries, white_xyY = calc_primaries_and_white(spd=msd)
    # print(primaries)
    # print(white_xyY)
    ds = DisplaySpectrum(msd=msd)

    rgb_gain = generate_color_checker_rgb_value()
    print(rgb_gain)
