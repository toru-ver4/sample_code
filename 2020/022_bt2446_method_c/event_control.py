# -*- coding: utf-8 -*-
"""
Event controller for 'parameter adjustment tool'.
=================================================

"""

# import standard libraries
import os

# import third-party libraries
import cv2
import numpy as np

# import my libraries
from key_names import KeyNames
import bt2446_method_c as bmc
import bt2446_plot as btp

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

kns = KeyNames()


class EventControl():
    """
    event controller for parameter_adjustment_tool.
    """
    def __init__(self, window, fig_agg, ax_lines):
        self.window = window
        self.fig_agg = fig_agg
        self.tonecurve = ax_lines[0]
        self.hdr_ip_line = ax_lines[1]
        self.hdr_ref_line = ax_lines[2]
        self.handler = {
            kns.hdr_ref_white_spin: self.update_hdr_ref_white_by_spin,
            kns.hdr_ref_white_slider: self.update_hdr_ref_white_by_slider,
            kns.hdr_peak_white_spin: self.update_hdr_peak_white_by_spin,
            kns.hdr_peak_white_slider: self.update_hdr_peak_white_by_slider,
            kns.cross_talk_alpha_spin: self.update_alpha_by_spin,
            kns.cross_talk_alpha_slider: self.update_alpha_by_slider,
            kns.cross_talk_sigma_spin: self.update_sigma_by_spin,
            kns.cross_talk_sigma_slider: self.update_sigma_by_slider,
            kns.k1_spin: self.update_k1_by_spin,
            kns.k1_slider: self.update_k1_by_slider,
            kns.k3_spin: self.update_k3_by_spin,
            kns.k3_slider: self.update_k3_by_slider,
            kns.sdr_ip_spin: self.update_sdr_ip_by_spin,
            kns.sdr_ip_slider: self.update_sdr_ip_by_slider
        }

    def get_handler(self):
        return self.handler

    def update_tonecurve(self, values):
        k1 = values[kns.k1_slider]
        k3 = values[kns.k3_slider]
        y_sdr_ip = values[kns.sdr_ip_slider]
        y_hdr_ref = values[kns.hdr_ref_white_slider]
        x = np.linspace(0, 10000, 1024)
        y = bmc.bt2446_method_c_tonemapping(
            x, k1=k1, k3=k3, y_sdr_ip=y_sdr_ip, y_hdr_ref=y_hdr_ref)
        y_hdr_ip, y_sdr_wp, k2, k4 = bmc.calc_tonemapping_parameters(
            k1=k1, k3=k3, y_sdr_ip=y_sdr_ip, y_hdr_ref=y_hdr_ref)
        self.tonecurve.set_data(x, y)
        self.hdr_ip_line.set_data(
            *btp.calc_hdr_ip_sub_line(y_hdr_ip=y_hdr_ip, k1=k1))
        self.hdr_ref_line.set_data(
            *btp.calc_hdr_ref_sub_line(y_hdr_ref=y_hdr_ref, y_sdr_wp=y_sdr_wp))
        self.fig_agg.draw()
        self.window[kns.info_y_hdr_ip].update(f"{y_hdr_ip:.1f}")
        self.window[kns.info_y_sdr_wp].update(f"{y_sdr_wp:.1f}")
        print(f"{y_sdr_wp:.1f}")

    def update_hdr_peak_white_by_spin(self, values):
        self.window[kns.hdr_peak_white_slider].update(
            values[kns.hdr_peak_white_spin])
        self.update_tonecurve(values)

    def update_hdr_peak_white_by_slider(self, values):
        self.window[kns.hdr_peak_white_spin].update(
            values[kns.hdr_peak_white_slider])
        self.update_tonecurve(values)

    def update_hdr_ref_white_by_spin(self, values):
        self.window[kns.hdr_ref_white_slider].update(
            values[kns.hdr_ref_white_spin])
        self.update_tonecurve(values)

    def update_hdr_ref_white_by_slider(self, values):
        self.window[kns.hdr_ref_white_spin].update(
            values[kns.hdr_ref_white_slider])
        self.update_tonecurve(values)

    def update_alpha_by_spin(self, values):
        self.window[kns.cross_talk_alpha_slider].update(
            values[kns.cross_talk_alpha_spin])
        self.update_tonecurve(values)

    def update_alpha_by_slider(self, values):
        self.window[kns.cross_talk_alpha_spin].update(
            values[kns.cross_talk_alpha_slider])
        self.update_tonecurve(values)

    def update_sigma_by_spin(self, values):
        self.window[kns.cross_talk_sigma_slider].update(
            values[kns.cross_talk_sigma_spin])
        self.update_tonecurve(values)

    def update_sigma_by_slider(self, values):
        self.window[kns.cross_talk_sigma_spin].update(
            values[kns.cross_talk_sigma_slider])
        self.update_tonecurve(values)

    def update_k1_by_spin(self, values):
        self.window[kns.k1_slider].update(values[kns.k1_spin])
        self.update_tonecurve(values)

    def update_k1_by_slider(self, values):
        self.window[kns.k1_spin].update(values[kns.k1_slider])
        self.update_tonecurve(values)

    def update_k3_by_spin(self, values):
        self.window[kns.k3_slider].update(values[kns.k3_spin])
        self.update_tonecurve(values)

    def update_k3_by_slider(self, values):
        self.window[kns.k3_spin].update(values[kns.k3_slider])
        self.update_tonecurve(values)

    def update_sdr_ip_by_spin(self, values):
        self.window[kns.sdr_ip_slider].update(values[kns.sdr_ip_spin])
        self.update_tonecurve(values)

    def update_sdr_ip_by_slider(self, values):
        self.window[kns.sdr_ip_spin].update(values[kns.sdr_ip_slider])
        self.update_tonecurve(values)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
