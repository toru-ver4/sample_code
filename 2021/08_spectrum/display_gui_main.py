# -*- coding: utf-8 -*-
"""
simulation
"""

# import standard libraries
import os
import sys

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt

from PySide2.QtWidgets import QApplication, QHBoxLayout, QWidget, QSlider,\
    QLabel, QVBoxLayout, QGridLayout
from PySide2.QtCore import Qt
from PySide2.QtGui import QPixmap, QImage, QPalette, QColor, QFont
from matplotlib.backends.backend_qt5agg\
    import FigureCanvasQTAgg as FigureCanvas
from colour.models import RGB_COLOURSPACE_BT709

# import my libraries
import test_pattern_generator2 as tpg
import plot_utility as pu
import transfer_functions as tf
import color_space as cs
from spectrum_calculation import calc_illuminant_d_spectrum,\
    get_cie_2_1931_cmf, get_cie_2_2012_cmf, calc_linear_rgb_from_spectrum,\
    REFRECT_100P_SD, get_color_checker_large_xyz_of_d65,\
    convert_color_checker_linear_rgb_from_d65,\
    plot_color_checker_image, load_color_checker_spectrum,\
    calc_color_temp_after_spectrum_rendering, DisplaySpectralDistribution,\
    calc_xyY_from_single_spectrum, calc_gain_for_white_point_adjust_rgb,\
    calc_primary_xyY_and_white_xyY, COLOR_TEMP_15K, COLOR_TEMP_4K,\
    calc_gobal_gain_for_bright_adjust


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


class TyBasicSlider(QWidget):
    def __init__(
            self, int_float_rate=10, default=2.2, min_val=0.1, max_val=3.0):
        """
        Parameters
        ----------
        int_float_rate : float
            int to float rate.
        default : float
            default value
        min_val : float
            minimum value
        max_val : float
            maximum value

        Example
        -------
        >>> slider = TyBasicSlider(
        ...     int_float_rate=10, default=2.2, min_val=0.1, max_val=3.0)
        """
        super().__init__()
        self.int_float_rate = int_float_rate
        self.defalut_value = default
        self.slider = QSlider(orientation=Qt.Horizontal)
        self.slider.setMinimum(int(min_val * self.int_float_rate))
        self.slider.setMaximum(int(max_val * self.int_float_rate))
        self.slider.setValue(int(self.defalut_value * self.int_float_rate))
        self.slider.setTickPosition(QSlider.TicksBelow)

    def set_value(self, value=2.2):
        self.slider.setValue(int(value * self.int_float_rate))

    def get_value(self):
        return self.slider.value() / self.int_float_rate

    def get_widget(self):
        return self.slider

    def get_default(self):
        return self.defalut_value

    def set_slot(self, slot_func):
        self.slot_func = slot_func
        self.slider.valueChanged.connect(self.slot_func)


class TyBasicLabel(QWidget):
    def __init__(
            self, default=2.2, prefix="", suffix="",
            font='Noto Sans Mono CJK JP', font_size=12,
            font_weight=QFont.Medium):
        super().__init__()
        self.label = QLabel(f"{prefix} {str(default)} {suffix}")
        self.label.setFont(
            QFont(font, font_size, font_weight))

        self.prefix = prefix
        self.suffix = suffix
        self.internal_value = 1.0

    def get_widget(self):
        return self.label

    def set_label(self, value):
        self.internal_value = value
        self.label.setText(
            f"{self.prefix} {int(self.internal_value):4d} {self.suffix}")

    def get_label(self):
        return float(self.internal_value)


class TyxyYLabel(TyBasicLabel):
    def __init__(
            self, default=2.2, prefix="", suffix="",
            font='Noto Sans Mono CJK JP', font_size=12,
            font_weight=QFont.Medium):
        super().__init__(
                default=default, prefix=prefix, font='Noto Sans Mono CJK JP',
                font_size=12, font_weight=QFont.Medium)

    def set_label(self, value):
        self.internal_value = value
        label_txt = f"({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})"
        self.label.setText(
            f"{self.prefix} {label_txt} {self.suffix}")


class TyDxyLabel(TyBasicLabel):
    def __init__(
            self, default=2.2, prefix="", suffix="",
            font='Noto Sans Mono CJK JP', font_size=12,
            font_weight=QFont.Medium):
        super().__init__(
                default=default, prefix=prefix, font='Noto Sans Mono CJK JP',
                font_size=12, font_weight=QFont.Medium)

    def set_label(self, value):
        self.internal_value = value
        self.label.setText(
            f"{self.prefix} {value:.3f} {self.suffix}")


class LayoutControl():
    def __init__(self, parent) -> None:
        # self.base_layout = QHBoxLayout(parent)
        self.base_layout = QGridLayout()
        parent.setLayout(self.base_layout)

    def set_mpl_layout(
            self, canvas, chromaticity_diagram_canvas,
            r_mean_label, g_mean_label, b_mean_label,
            r_dist_label, g_dist_label, b_dist_label,
            r_gain_label, g_gain_label, b_gain_label,
            r_mean_slider, g_mean_slider, b_mean_slider,
            r_dist_slider, g_dist_slider, b_dist_slider,
            color_temp_slider, color_temp_label,
            cie1931_xyY_label, cie2012_xyY_label,
            cie1931_dx_label, cie1931_dy_label):
        r_layout = QHBoxLayout()
        g_layout = QHBoxLayout()
        b_layout = QHBoxLayout()
        color_temp_layout = QHBoxLayout()

        r_layout.addWidget(r_mean_label.get_widget())
        r_layout.addWidget(r_mean_slider.get_widget())
        r_layout.addWidget(r_dist_label.get_widget())
        r_layout.addWidget(r_dist_slider.get_widget())
        r_layout.addWidget(r_gain_label.get_widget())

        g_layout.addWidget(g_mean_label.get_widget())
        g_layout.addWidget(g_mean_slider.get_widget())
        g_layout.addWidget(g_dist_label.get_widget())
        g_layout.addWidget(g_dist_slider.get_widget())
        g_layout.addWidget(g_gain_label.get_widget())

        b_layout.addWidget(b_mean_label.get_widget())
        b_layout.addWidget(b_mean_slider.get_widget())
        b_layout.addWidget(b_dist_label.get_widget())
        b_layout.addWidget(b_dist_slider.get_widget())
        b_layout.addWidget(b_gain_label.get_widget())

        color_temp_layout.addWidget(color_temp_label.get_widget())
        color_temp_layout.addWidget(color_temp_slider.get_widget())

        mpl_layout = QVBoxLayout()
        mpl_layout.addWidget(canvas.get_widget())
        mpl_layout.addLayout(r_layout)
        mpl_layout.addLayout(g_layout)
        mpl_layout.addLayout(b_layout)
        mpl_layout.addLayout(color_temp_layout)

        cie_xyY_layout = QHBoxLayout()
        cie_xyY_layout.addWidget(cie1931_xyY_label.get_widget())
        cie_xyY_layout.addWidget(cie2012_xyY_label.get_widget())
        cie_dxy_layout = QHBoxLayout()
        cie_dxy_layout.addWidget(cie1931_dx_label.get_widget())
        cie_dxy_layout.addWidget(cie1931_dy_label.get_widget())

        chroma_diagram_layout = QVBoxLayout()
        chroma_diagram_layout.addWidget(
            chromaticity_diagram_canvas.get_widget())
        chroma_diagram_layout.addLayout(cie_xyY_layout)
        chroma_diagram_layout.addLayout(cie_dxy_layout)

        self.base_layout.addLayout(mpl_layout, 0, 0)
        self.base_layout.addLayout(chroma_diagram_layout, 0, 1)

    def set_color_patch_layout(self, color_patch):
        patch_layout = QVBoxLayout()
        patch_layout.addWidget(color_patch.get_widget())
        self.base_layout.addLayout(patch_layout, 0, 1)


class WindowColorControl():
    def __init__(self, parent) -> None:
        self.parent = parent

    def set_bg_color(self, color=[0.15, 0.15, 0.15]):
        palette = self.parent.palette()
        color_value = [int(x * 255) for x in color]
        palette.setColor(QPalette.Window, QColor(*color_value))
        self.parent.setPalette(palette)
        self.parent.setAutoFillBackground(True)


class ColorPatchImage():
    def __init__(self, width=256, height=256, color_temp_default=6500):
        self.label = QLabel()
        self.width = width
        self.height = height
        self.image_change(color_temp=color_temp_default)

    def ndarray_to_qimage(self, img):
        """
        Parameters
        ----------
        img : ndarray(float)
            image data. data range is from 0.0 to 1.0
        """
        self.uint8_img = np.uint8(np.round(np.clip(img, 0.0, 1.0) * 255))
        height, width = self.uint8_img.shape[:2]
        self.qimg = QImage(
            self.uint8_img.data, width, height, QImage.Format_RGB888)

        return self.qimg

    def image_change(self, color_temp=6500):
        src_sd = calc_illuminant_d_spectrum(color_temp=color_temp)
        ref_sd = REFRECT_100P_SD
        cmfs = get_cie_2_1931_cmf()

        rgb_linear = calc_linear_rgb_from_spectrum(
            src_sd=src_sd, ref_sd=ref_sd, cmfs=cmfs,
            color_space=RGB_COLOURSPACE_BT709).reshape((1, 1, 3))
        print(f"rgb_linear={rgb_linear}")
        rgb_srgb = tf.oetf(np.clip(rgb_linear, 0.0, 1.0), tf.SRGB)
        print(f"rgb_srgb={rgb_srgb}")

        self.qimg = self.ndarray_to_qimage(
            np.ones((self.height, self.width, 3), dtype=np.uint8) * rgb_srgb)
        self.label.setPixmap(QPixmap.fromImage(self.qimg))

    def get_widget(self):
        return self.label


class ColorCheckerImage():
    def __init__(self, width=540, height=360, color_temp_default=6504):
        self.label = QLabel()
        self.width = width
        self.height = height
        self.d65_color_checker_xyz = get_color_checker_large_xyz_of_d65(
            color_temp=color_temp_default)
        self.cmfs = get_cie_2_1931_cmf()
        self.cc_ref_sd = load_color_checker_spectrum()
        self.image_change(color_temp=color_temp_default)

    def ndarray_to_qimage(self, img):
        """
        Parameters
        ----------
        img : ndarray(float)
            image data. data range is from 0.0 to 1.0
        """
        self.uint8_img = np.uint8(np.round(np.clip(img, 0.0, 1.0) * 255))
        height, width = self.uint8_img.shape[:2]
        self.qimg = QImage(
            self.uint8_img.data, width, height, QImage.Format_RGB888)

        return self.qimg

    def get_widget(self):
        return self.label

    def image_change(self, color_temp=6504):
        src_sd = calc_illuminant_d_spectrum(color_temp)
        linear_rgb = calc_linear_rgb_from_spectrum(
            src_sd=src_sd, ref_sd=self.cc_ref_sd, cmfs=self.cmfs,
            color_space=RGB_COLOURSPACE_BT709)
        rgb_srgb = tf.oetf(np.clip(linear_rgb, 0.0, 1.0), tf.SRGB)
        result_xy = calc_color_temp_after_spectrum_rendering(
            src_sd=src_sd, cmfs=self.cmfs)
        linear_rgb_mtx = convert_color_checker_linear_rgb_from_d65(
            d65_color_checker_xyz=self.d65_color_checker_xyz,
            dst_white=result_xy, color_space=RGB_COLOURSPACE_BT709)
        rgb_srgb_mtx = tf.oetf(np.clip(linear_rgb_mtx, 0.0, 1.0), tf.SRGB)
        color_checker_img = plot_color_checker_image(
            rgb=rgb_srgb, rgb2=rgb_srgb_mtx, size=(self.width, self.height),
            block_size=1/4.5)

        self.qimg = self.ndarray_to_qimage(color_checker_img)
        self.label.setPixmap(QPixmap.fromImage(self.qimg))


class TySpectrumPlot():
    def __init__(
            self, figsize=(10, 8), default_temp=6500):
        super().__init__()
        self.fig, self.ax1 = pu.plot_1_graph(
            fontsize=14,
            figsize=figsize,
            graph_title="Spectral power distribution",
            graph_title_size=None,
            xlabel="Wavelength [nm]", ylabel="???",
            axis_label_size=12,
            legend_size=12,
            xlim=[330, 860],
            ylim=[-0.05, 2.0],
            xtick=[350 + x * 50 for x in range(11)],
            ytick=None,
            xtick_size=12, ytick_size=12,
            linewidth=3,
            minor_xtick_num=None,
            minor_ytick_num=None,
            return_figure=True)
        self.plot_init_plot(sample_num=256, default_temp=6500)
        self.canvas = FigureCanvas(self.fig)
        # self.plot_obj.figure.canvas.draw()

    def plot_init_plot(self, sample_num, default_temp):
        illuminant_d = calc_illuminant_d_spectrum(color_temp=default_temp)
        self.illuminant_x = illuminant_d.wavelengths
        self.illuminant_y = illuminant_d.values
        self.illuminant_line, = self.ax1.plot(
            self.illuminant_x, self.illuminant_y, color=[0.2, 0.2, 0.2],
            label="SPD of illuminant D")

        cmf = get_cie_2_1931_cmf()
        self.cmf_x = cmf.wavelengths
        self.cmf_r = cmf.values[..., 0]
        self.cmf_g = cmf.values[..., 1]
        self.cmf_b = cmf.values[..., 2]
        self.line_cmf_r = self.ax1.plot(
            self.cmf_x, self.cmf_r, '--', color=pu.RED,
            label="Color matching function(R)")
        self.line_cmf_g = self.ax1.plot(
            self.cmf_x, self.cmf_g, '--', color=pu.GREEN,
            label="Color matching function(G)")
        self.line_cmf_b = self.ax1.plot(
            self.cmf_x, self.cmf_b, '--', color=pu.BLUE,
            label="Color matching function(B)")
        plt.legend(loc='upper right')

    def get_widget(self):
        return self.canvas

    def update_plot(self, x, y):
        self.illuminant_line.set_data(x, y)
        self.illuminant_line.figure.canvas.draw()


class DisplaySpectrumPlot():
    def __init__(
            self, r_mean, r_dist, r_gain,
            g_mean, g_dist, g_gain, b_mean, b_dist, b_gain, figsize=(10, 6)):
        super().__init__()
        self.figsize = figsize
        self.cmfs = get_cie_2_1931_cmf()
        self.cmfs2012 = get_cie_2_2012_cmf()
        self.update_spectrum(
            r_mean=r_mean, r_dist=r_dist, r_gain=r_gain,
            g_mean=g_mean, g_dist=g_dist, g_gain=g_gain,
            b_mean=b_mean, b_dist=b_dist, b_gain=b_gain)
        self.init_plot()

    def update_spectrum(
            self, r_mean, r_dist, r_gain,
            g_mean, g_dist, g_gain, b_mean, b_dist, b_gain):
        param_dict = dict(
            wavelengths=np.arange(360, 831),
            r_mean=r_mean, r_dist=r_dist, r_gain=r_gain,
            g_mean=g_mean, g_dist=g_dist, g_gain=g_gain,
            b_mean=b_mean, b_dist=b_dist, b_gain=b_gain)
        self.display_sd_obj = DisplaySpectralDistribution(**param_dict)
        self.display_w_sd = self.display_sd_obj.get_wrgb_sd_array()[0]

    def init_plot(self):
        self.fig, ax1 = pu.plot_1_graph(
            fontsize=14,
            figsize=self.figsize,
            graph_title="Spectral power distribution",
            graph_title_size=None,
            xlabel="Wavelength [nm]", ylabel="???",
            axis_label_size=None,
            legend_size=12,
            xlim=[340, 750],
            ylim=[-0.05, 5],
            xtick=None,
            ytick=None,
            xtick_size=None, ytick_size=None,
            linewidth=3,
            return_figure=True)
        self.display_sd_line, = ax1.plot(
            self.display_w_sd.wavelengths, self.display_w_sd.values, '-',
            color=(0.1, 0.1, 0.1), label="Display (W=R+G+B)")
        ax1.plot(
            self.cmfs.wavelengths, self.cmfs.values[..., 0], '-',
            color=pu.RED, label="CIE 1931 2 CMF(R)", lw=1.5)
        ax1.plot(
            self.cmfs.wavelengths, self.cmfs.values[..., 1], '-',
            color=pu.GREEN, label="CIE 1931 2 CMF(G)", lw=1.5)
        ax1.plot(
            self.cmfs.wavelengths, self.cmfs.values[..., 2], '-',
            color=pu.BLUE, label="CIE 1931 2 CMF(B)", lw=1.5)

        ax1.plot(
            self.cmfs2012.wavelengths, self.cmfs2012.values[..., 0], '--',
            color=pu.RED, label="CIE 2012 2 CMF(R)", lw=1.5)
        ax1.plot(
            self.cmfs2012.wavelengths, self.cmfs2012.values[..., 1], '--',
            color=pu.GREEN, label="CIE 2012 2 CMF(G)", lw=1.5)
        ax1.plot(
            self.cmfs2012.wavelengths, self.cmfs2012.values[..., 2], '--',
            color=pu.BLUE, label="CIE 2012 2 CMF(B)", lw=1.5)

        plt.legend(loc='upper right')
        self.canvas = FigureCanvas(self.fig)

    def update_plot(
            self, r_mean, r_dist, r_gain,
            g_mean, g_dist, g_gain, b_mean, b_dist, b_gain):
        self.update_spectrum(
            r_mean=r_mean, r_dist=r_dist, r_gain=r_gain,
            g_mean=g_mean, g_dist=g_dist, g_gain=g_gain,
            b_mean=b_mean, b_dist=b_dist, b_gain=b_gain)
        self.display_sd_line.set_data(
            self.display_w_sd.wavelengths, self.display_w_sd.values)
        self.display_sd_line.figure.canvas.draw()

    def get_widget(self):
        return self.canvas

    def get_display_sd_obj(self):
        return self.display_sd_obj


class ChromaticityDiagramPlot():
    def __init__(self, display_sd_obj):
        super().__init__()
        primaries, white = calc_primary_xyY_and_white_xyY(display_sd_obj)
        self.xmin = 0.0
        self.xmax = 0.8
        self.ymin = 0.0
        self.ymax = 0.9
        self.xy_image = tpg.get_chromaticity_image(
            xmin=self.xmin, xmax=self.xmax,
            ymin=self.ymin, ymax=self.ymax,
            bg_color=0.8, samples=384)
        self.plot_diagram_all(primaries, white)

    def plot_diagram_all(self, primaries, white):
        rate = 1
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        cmf_xy = tpg._get_cmfs_xy()
        bt709_gamut, _ = tpg.get_primaries(name=cs.BT709)
        bt2020_gamut, _ = tpg.get_primaries(name=cs.BT2020)
        dci_p3_gamut, _ = tpg.get_primaries(name=cs.P3_D65)
        xlim = (min(0, xmin), max(0.8, xmax))
        ylim = (min(0, ymin), max(0.9, ymax))
        self.fig, ax1 = pu.plot_1_graph(
            fontsize=16 * rate,
            figsize=((xmax - xmin) * 10 * rate, (ymax - ymin) * 10 * rate),
            graph_title="CIE1931 Chromaticity Diagram",
            graph_title_size=None,
            xlabel=None, ylabel=None,
            axis_label_size=None,
            legend_size=12 * rate,
            xlim=xlim, ylim=ylim,
            xtick=[x * 0.1 + xmin for x in
                   range(int((xlim[1] - xlim[0])/0.1) + 1)],
            ytick=[x * 0.1 + ymin for x in
                   range(int((ylim[1] - ylim[0])/0.1) + 1)],
            xtick_size=17 * rate,
            ytick_size=17 * rate,
            linewidth=4 * rate,
            minor_xtick_num=2,
            minor_ytick_num=2,
            return_figure=True)
        ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
        ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
                 '-k', lw=3.5*rate, label=None)
        ax1.plot(
            bt709_gamut[:, 0], bt709_gamut[:, 1],
            c=pu.SKY, label="BT.709", lw=1.5*rate, alpha=0.8)
        ax1.plot(
            bt2020_gamut[:, 0], bt2020_gamut[:, 1],
            c=pu.YELLOW, label="BT.2020", lw=1.5*rate, alpha=0.8)
        ax1.plot(
            dci_p3_gamut[:, 0], dci_p3_gamut[:, 1],
            c=pu.BLUE, label="DCI-P3", lw=1.5*rate, alpha=0.8)
        self.display_p_line, = ax1.plot(
            primaries[:, 0], primaries[:, 1],
            c='k', label="Display device", lw=2.75*rate)
        self.display_p_point, = ax1.plot(
            primaries[:, 0], primaries[:, 1], 's', c=(0.9, 0.9, 0.9), ms=8,
            mec=(0.2, 0.2, 0.2))
        ax1.plot(
            tpg.D65_WHITE[0], tpg.D65_WHITE[1], 'x', c=pu.RED, label="6504 K",
            ms=10, mew=3)
        ax1.plot(
            COLOR_TEMP_15K[0], COLOR_TEMP_15K[1], '+', c=pu.RED,
            label="15000 K", ms=12, mew=3)
        ax1.plot(
            COLOR_TEMP_4K[0], COLOR_TEMP_4K[1], 'o', c=pu.RED,
            label="4000 K", ms=8)
        self.display_w_line, = ax1.plot(
            white[0], white[1], 'x', c='k', label="White point", ms=10, mew=3)
        ax1.imshow(self.xy_image, extent=(xmin, xmax, ymin, ymax))
        plt.legend(loc='upper right')
        self.canvas = FigureCanvas(self.fig)

    def update_plot(self, display_sd_obj):
        primaries, white = calc_primary_xyY_and_white_xyY(display_sd_obj)
        self.display_p_line.set_data(primaries[:, 0], primaries[:, 1])
        self.display_p_point.set_data(primaries[:, 0], primaries[:, 1])
        self.display_w_line.set_data(white[0], white[1])

        self.canvas.draw()

    def get_widget(self):
        return self.canvas


class EventControl():
    def __init__(self) -> None:
        pass

    def set_display_sd_slider_event(
            self, display_sd_canvas, chromaticity_diagram_canvas,
            r_mean_label, g_mean_label, b_mean_label,
            r_dist_label, g_dist_label, b_dist_label,
            r_gain_label, g_gain_label, b_gain_label,
            r_mean_slider, g_mean_slider, b_mean_slider,
            r_dist_slider, g_dist_slider, b_dist_slider,
            color_temp_slider, color_temp_label,
            cie1931_xyY_label, cie2012_xyY_label,
            cie1931_dx_label, cie1931_dy_label):
        self.display_sd_canvas = display_sd_canvas
        self.chromaticity_diagram_canvas = chromaticity_diagram_canvas
        self.r_mean_label = r_mean_label
        self.g_mean_label = g_mean_label
        self.b_mean_label = b_mean_label
        self.r_dist_label = r_dist_label
        self.g_dist_label = g_dist_label
        self.b_dist_label = b_dist_label
        self.r_gain_label = r_gain_label
        self.g_gain_label = g_gain_label
        self.b_gain_label = b_gain_label
        self.r_mean_slider = r_mean_slider
        self.g_mean_slider = g_mean_slider
        self.b_mean_slider = b_mean_slider
        self.r_dist_slider = r_dist_slider
        self.g_dist_slider = g_dist_slider
        self.b_dist_slider = b_dist_slider
        self.color_temp_slider = color_temp_slider
        self.color_temp_label = color_temp_label

        self.r_mean_slider.set_slot(self.display_sd_slider_event)
        self.g_mean_slider.set_slot(self.display_sd_slider_event)
        self.b_mean_slider.set_slot(self.display_sd_slider_event)
        self.r_dist_slider.set_slot(self.display_sd_slider_event)
        self.g_dist_slider.set_slot(self.display_sd_slider_event)
        self.b_dist_slider.set_slot(self.display_sd_slider_event)
        self.color_temp_slider.set_slot(self.display_sd_slider_event)

        self.cie1931_xyY_label = cie1931_xyY_label
        self.cie2012_xyY_label = cie2012_xyY_label
        self.cie1931_dx_label = cie1931_dx_label
        self.cie1931_dy_label = cie1931_dy_label

        # gain is set automatically
        # self.r_gain_slider.set_slot(self.display_sd_slider_event)
        # self.g_gain_slider.set_slot(self.display_sd_slider_event)
        # self.b_gain_slider.set_slot(self.display_sd_slider_event)

    def display_sd_slider_event(self):
        r_mean_value = self.r_mean_slider.get_value()
        g_mean_value = self.g_mean_slider.get_value()
        b_mean_value = self.b_mean_slider.get_value()
        r_dist_value = self.r_dist_slider.get_value()
        g_dist_value = self.g_dist_slider.get_value()
        b_dist_value = self.b_dist_slider.get_value()
        color_temp = self.color_temp_slider.get_value()

        self.r_mean_label.set_label(r_mean_value)
        self.g_mean_label.set_label(g_mean_value)
        self.b_mean_label.set_label(b_mean_value)
        self.r_dist_label.set_label(r_dist_value)
        self.g_dist_label.set_label(g_dist_value)
        self.b_dist_label.set_label(b_dist_value)
        # r_gain_value = self.r_gain_label.get_label()
        g_gain_value = self.g_gain_label.get_label()
        # b_gain_value = self.b_gain_label.get_label()

        # pre calculation
        self.display_sd_canvas.update_spectrum(
            r_mean=r_mean_value, r_dist=r_dist_value, r_gain=g_gain_value,
            g_mean=g_mean_value, g_dist=g_dist_value, g_gain=g_gain_value,
            b_mean=b_mean_value, b_dist=b_dist_value, b_gain=g_gain_value)

        # calc r, g, b gain
        r_gain_value, g_gain_value, b_gain_value\
            = calc_gain_for_white_point_adjust_rgb(
                src_display_sd_obj=self.display_sd_canvas.get_display_sd_obj(),
                dst_color_temp=color_temp)

        # apply r, g, b gain
        self.display_sd_canvas.update_spectrum(
            r_mean=r_mean_value, r_dist=r_dist_value, r_gain=g_gain_value,
            g_mean=g_mean_value, g_dist=g_dist_value, g_gain=g_gain_value,
            b_mean=b_mean_value, b_dist=b_dist_value, b_gain=g_gain_value)

        # calc global gain
        global_gain = calc_gobal_gain_for_bright_adjust(
            display_sd_obj=self.display_sd_canvas.get_display_sd_obj())
        r_gain_value = r_gain_value * global_gain
        g_gain_value = g_gain_value * global_gain
        b_gain_value = b_gain_value * global_gain

        self.r_gain_label.set_label(r_gain_value)
        self.g_gain_label.set_label(g_gain_value)
        self.b_gain_label.set_label(b_gain_value)
        self.color_temp_label.set_label(color_temp)

        # main update
        self.display_sd_canvas.update_plot(
            r_mean=r_mean_value, r_dist=r_dist_value, r_gain=r_gain_value,
            g_mean=g_mean_value, g_dist=g_dist_value, g_gain=g_gain_value,
            b_mean=b_mean_value, b_dist=b_dist_value, b_gain=b_gain_value)

        # calc xyY
        display_sd_obj = self.display_sd_canvas.get_display_sd_obj()
        w_display_sd = display_sd_obj.get_wrgb_sd_array()[0]
        cmfs_1931 = get_cie_2_1931_cmf()
        cmfs_2012 = get_cie_2_2012_cmf()
        xyY_1931 = calc_xyY_from_single_spectrum(
            src_sd=REFRECT_100P_SD, ref_sd=w_display_sd, cmfs=cmfs_1931)
        xyY_2012 = calc_xyY_from_single_spectrum(
            src_sd=REFRECT_100P_SD, ref_sd=w_display_sd, cmfs=cmfs_2012)
        self.cie1931_xyY_label.set_label(xyY_1931)
        self.cie2012_xyY_label.set_label(xyY_2012)
        self.cie1931_dx_label.set_label((xyY_1931 - xyY_2012)[0])
        self.cie1931_dy_label.set_label((xyY_1931 - xyY_2012)[1])

        self.chromaticity_diagram_canvas.update_plot(
            display_sd_obj=self.display_sd_canvas.get_display_sd_obj())


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1280, 720)
        r_gain_default = 50
        g_gain_default = 50
        b_gain_default = 50

        # background color
        window_color = WindowColorControl(parent=self)
        window_color.set_bg_color(color=[0.8, 0.8, 0.8])

        # layout
        layout = LayoutControl(self)

        # object for widget
        r_mean_slider = TyBasicSlider(
            int_float_rate=1/5, default=625, min_val=525, max_val=725)
        g_mean_slider = TyBasicSlider(
            int_float_rate=1/5, default=530, min_val=430, max_val=630)
        b_mean_slider = TyBasicSlider(
            int_float_rate=1/5, default=460, min_val=360, max_val=560)
        r_dist_slider = TyBasicSlider(
            int_float_rate=1, default=25, min_val=1, max_val=60)
        g_dist_slider = TyBasicSlider(
            int_float_rate=1, default=25, min_val=1, max_val=60)
        b_dist_slider = TyBasicSlider(
            int_float_rate=1, default=25, min_val=1, max_val=60)
        color_temp_slider = TyBasicSlider(
            int_float_rate=1/100, default=6500, min_val=4000, max_val=15000)

        r_mean_label = TyBasicLabel(
            default=r_mean_slider.get_default(),
            prefix="R_mean:", suffix="[nm]")
        g_mean_label = TyBasicLabel(
            default=g_mean_slider.get_default(),
            prefix="G_mean:", suffix="[nm]")
        b_mean_label = TyBasicLabel(
            default=b_mean_slider.get_default(),
            prefix="B_mean:", suffix="[nm]")
        r_dist_label = TyBasicLabel(
            default=r_dist_slider.get_default(), prefix=" R_sd:", suffix="")
        g_dist_label = TyBasicLabel(
            default=g_dist_slider.get_default(), prefix=" G_sd:", suffix="")
        b_dist_label = TyBasicLabel(
            default=b_dist_slider.get_default(), prefix=" B_sd:", suffix="")
        r_gain_label = TyBasicLabel(default=r_gain_default, prefix=" R_gain:")
        g_gain_label = TyBasicLabel(default=g_gain_default, prefix=" G_gain:")
        b_gain_label = TyBasicLabel(default=b_gain_default, prefix=" B_gain:")
        color_temp_label = TyBasicLabel(
            default=color_temp_slider.get_default(), suffix="[K]")
        cie1931_xyY_label = TyxyYLabel(default=0.0, prefix="CIE1931 xyY=")
        cie2012_xyY_label = TyxyYLabel(default=0.0, prefix="CIE2012 xyY=")
        cie1931_dx_label = TyDxyLabel(default=0.0, prefix="Δx=")
        cie1931_dy_label = TyDxyLabel(default=0.0, prefix="Δy=")

        # spectrum_plot = TySpectrumPlot(
        #     default_temp=white_slider.get_default(), figsize=(10, 6))
        display_sd_plot = DisplaySpectrumPlot(
            r_mean=r_mean_slider.get_default(),
            r_dist=r_dist_slider.get_default(),
            r_gain=r_gain_default,
            g_mean=g_mean_slider.get_default(),
            g_dist=g_dist_slider.get_default(),
            g_gain=g_gain_default,
            b_mean=b_mean_slider.get_default(),
            b_dist=b_dist_slider.get_default(),
            b_gain=b_gain_default, figsize=(10, 6))

        chromaticity_diagram = ChromaticityDiagramPlot(
            display_sd_obj=display_sd_plot.get_display_sd_obj())

        # set slot
        self.event_control = EventControl()
        # self.event_control.set_white_slider_event(
        #     white_slider=white_slider, white_label=white_label,
        #     spectrum_plot=spectrum_plot, patch_img=color_checkr_img)
        self.event_control.set_display_sd_slider_event(
            display_sd_canvas=display_sd_plot,
            chromaticity_diagram_canvas=chromaticity_diagram,
            r_mean_label=r_mean_label, g_mean_label=g_mean_label,
            b_mean_label=b_mean_label, r_dist_label=r_dist_label,
            g_dist_label=g_dist_label, b_dist_label=b_dist_label,
            r_gain_label=r_gain_label, g_gain_label=g_gain_label,
            b_gain_label=b_gain_label, r_mean_slider=r_mean_slider,
            g_mean_slider=g_mean_slider, b_mean_slider=b_mean_slider,
            r_dist_slider=r_dist_slider, g_dist_slider=g_dist_slider,
            b_dist_slider=b_dist_slider, color_temp_slider=color_temp_slider,
            color_temp_label=color_temp_label,
            cie1931_xyY_label=cie1931_xyY_label,
            cie2012_xyY_label=cie2012_xyY_label,
            cie1931_dx_label=cie1931_dx_label,
            cie1931_dy_label=cie1931_dy_label)

        # set layout
        layout.set_mpl_layout(
            canvas=display_sd_plot,
            chromaticity_diagram_canvas=chromaticity_diagram,
            r_mean_label=r_mean_label, g_mean_label=g_mean_label,
            b_mean_label=b_mean_label, r_dist_label=r_dist_label,
            g_dist_label=g_dist_label, b_dist_label=b_dist_label,
            r_gain_label=r_gain_label, g_gain_label=g_gain_label,
            b_gain_label=b_gain_label, r_mean_slider=r_mean_slider,
            g_mean_slider=g_mean_slider, b_mean_slider=b_mean_slider,
            r_dist_slider=r_dist_slider, g_dist_slider=g_dist_slider,
            b_dist_slider=b_dist_slider, color_temp_slider=color_temp_slider,
            color_temp_label=color_temp_label,
            cie1931_xyY_label=cie1931_xyY_label,
            cie2012_xyY_label=cie2012_xyY_label,
            cie1931_dx_label=cie1931_dx_label,
            cie1931_dy_label=cie1931_dy_label)


def main_func():
    app = QApplication([])
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
