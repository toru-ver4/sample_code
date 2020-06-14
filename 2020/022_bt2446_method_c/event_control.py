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
    def __init__(self, window, fig_agg, ax_lines, im_pro):
        self.window = window
        self.im_pro = im_pro
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
            kns.sdr_ip_slider: self.update_sdr_ip_by_slider,
            kns.load_images: self.load_init_images,
            kns.update: self.update_sdr_images
        }

    def get_handler(self):
        return self.handler

    def update_tonecurve(self, values):
        k1 = values[kns.k1_slider]
        k3 = values[kns.k3_slider]
        y_sdr_ip = values[kns.sdr_ip_slider]
        y_hdr_ref = values[kns.hdr_ref_white_slider]
        x = np.linspace(0, 10000, 1024)
        y = bmc.bt2446_method_c_tonemapping_core(
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

    def load_init_images(self, values):
        self.im_pro.read_and_concat_img()

        # raw を真ん中に表示
        concat_raw_img = self.im_pro.get_concat_raw_image()
        concat_raw_img_8bit = self.im_pro.conv_8bit_int(concat_raw_img)
        self.window[kns.img_tp_raw].update(
            data=self.im_pro.conv_io_stream(
                self.im_pro.extract_image(concat_raw_img_8bit, 0)))
        self.window[kns.img_low_raw].update(
            data=self.im_pro.conv_io_stream(
                self.im_pro.extract_image(concat_raw_img_8bit, 1)))
        self.window[kns.img_mid_raw].update(
            data=self.im_pro.conv_io_stream(
                self.im_pro.extract_image(concat_raw_img_8bit, 2)))
        self.window[kns.img_high_raw].update(
            data=self.im_pro.conv_io_stream(
                self.im_pro.extract_image(concat_raw_img_8bit, 3)))

        # colormap を右側に表示
        self.im_pro.make_colormap_image(turbo_peak_luminance=4000)
        color_map_img_8bit = self.im_pro.conv_8bit_int(
            self.im_pro.get_colormap_image())

        self.window[kns.img_tp_luminance].update(
            data=self.im_pro.conv_io_stream(
                self.im_pro.extract_image(color_map_img_8bit, 0)))
        self.window[kns.img_low_luminance].update(
            data=self.im_pro.conv_io_stream(
                self.im_pro.extract_image(color_map_img_8bit, 1)))
        self.window[kns.img_mid_luminance].update(
            data=self.im_pro.conv_io_stream(
                self.im_pro.extract_image(color_map_img_8bit, 2)))
        self.window[kns.img_high_luminance].update(
            data=self.im_pro.conv_io_stream(
                self.im_pro.extract_image(color_map_img_8bit, 3)))

    def update_sdr_images(self, values):
        alpha = values[kns.cross_talk_alpha_slider]
        sigma = values[kns.cross_talk_sigma_slider]
        hdr_ref_luminance = values[kns.hdr_ref_white_slider]
        hdr_peak_luminance = values[kns.hdr_peak_white_slider]
        k1 = values[kns.k1_slider]
        k3 = values[kns.k3_slider]
        y_sdr_ip = values[kns.sdr_ip_slider]
        sdr_image_8bit = self.im_pro.conv_8bit_int(
            self.im_pro.make_sdr_image(
                alpha=alpha, sigma=sigma, hdr_ref_luminance=hdr_ref_luminance,
                hdr_peak_luminance=hdr_peak_luminance,
                k1=k1, k3=k3, y_sdr_ip=y_sdr_ip, bt2407_gamut_mapping=True))

        self.window[kns.img_tp_mapping].update(
            data=self.im_pro.conv_io_stream(
                self.im_pro.extract_image(sdr_image_8bit, 0)))
        self.window[kns.img_low_mapping].update(
            data=self.im_pro.conv_io_stream(
                self.im_pro.extract_image(sdr_image_8bit, 1)))
        self.window[kns.img_mid_mapping].update(
            data=self.im_pro.conv_io_stream(
                self.im_pro.extract_image(sdr_image_8bit, 2)))
        self.window[kns.img_high_mapping].update(
            data=self.im_pro.conv_io_stream(
                self.im_pro.extract_image(sdr_image_8bit, 3)))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
