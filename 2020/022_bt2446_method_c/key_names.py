# -*- coding: utf-8 -*-
from typing import NamedTuple


class KeyNames(NamedTuple):
    update: str = 'submit_button'
    load_images: str = 'load_images'
    curve_plot: str = 'curve_plot'
    hdr_ref_white_spin: str = 'hdr_ref_white_spin'
    hdr_ref_white_slider: str = 'hdr_ref_white_slider'
    hdr_peak_white_spin: str = 'hdr_peak_white_spin'
    hdr_peak_white_slider: str = 'hdr_peak_white_slider'
    cross_talk_alpha_spin: str = 'cross_talk_alpha_spin'
    cross_talk_alpha_slider: str = 'cross_talk_alpha_slider'
    cross_talk_sigma_spin: str = 'cross_talk_sigma_spin'
    cross_talk_sigma_slider: str = 'cross_talk_sigma_slider'
    k1_spin: str = "k1_sin"
    k1_slider: str = "k1_slider"
    k3_spin: str = "k3_sin"
    k3_slider: str = "k3_slider"
    sdr_ip_spin: str = "sdr_ip_spin"
    sdr_ip_slider: str = "sdr_ip_slider"
    img_tp_luminance: str = "img_tp_luminance"
    img_low_luminance: str = "img_low_luminance"
    img_mid_luminance: str = "img_mid_luminance"
    img_high_luminance: str = "img_high_luminance"
    img_tp_raw: str = "imt_tp_raw"
    img_low_raw: str = "imt_low_raw"
    img_mid_raw: str = "imt_mid_raw"
    img_high_raw: str = "imt_high_raw"
    img_tp_mapping: str = "img_tp_mapping"
    img_low_mapping: str = "img_low_mapping"
    img_mid_mapping: str = "img_mid_mapping"
    img_high_mapping: str = "img_high_mapping"
    info_y_hdr_ip: str = "info_y_hdr_ip"
    info_y_sdr_wp: str = "info_y_sdr_ip"
