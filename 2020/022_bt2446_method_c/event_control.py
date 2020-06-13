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
    def __init__(self, window):
        self.window = window
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
            kns.k3_slider: self.update_k3_by_slider
        }

    def get_handler(self):
        return self.handler

    def update_hdr_peak_white_by_spin(self, values):
        self.window[kns.hdr_peak_white_slider].update(
            values[kns.hdr_peak_white_spin])

    def update_hdr_peak_white_by_slider(self, values):
        self.window[kns.hdr_peak_white_spin].update(
            values[kns.hdr_peak_white_slider])

    def update_hdr_ref_white_by_spin(self, values):
        self.window[kns.hdr_ref_white_slider].update(
            values[kns.hdr_ref_white_spin])

    def update_hdr_ref_white_by_slider(self, values):
        self.window[kns.hdr_ref_white_spin].update(
            values[kns.hdr_ref_white_slider])

    def update_alpha_by_spin(self, values):
        self.window[kns.cross_talk_alpha_slider].update(
            values[kns.cross_talk_alpha_spin])

    def update_alpha_by_slider(self, values):
        self.window[kns.cross_talk_alpha_spin].update(
            values[kns.cross_talk_alpha_slider])

    def update_sigma_by_spin(self, values):
        self.window[kns.cross_talk_sigma_slider].update(
            values[kns.cross_talk_sigma_spin])

    def update_sigma_by_slider(self, values):
        self.window[kns.cross_talk_sigma_spin].update(
            values[kns.cross_talk_sigma_slider])

    def update_k1_by_spin(self, values):
        self.window[kns.k1_slider].update(values[kns.k1_spin])

    def update_k1_by_slider(self, values):
        self.window[kns.k1_spin].update(values[kns.k1_slider])

    def update_k3_by_spin(self, values):
        self.window[kns.k3_slider].update(values[kns.k3_spin])

    def update_k3_by_slider(self, values):
        self.window[kns.k3_spin].update(values[kns.k3_slider])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
