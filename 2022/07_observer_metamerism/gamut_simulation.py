# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os
import sys
from tkinter import Y

# import third-party libraries
import numpy as np
from PySide2.QtWidgets import QApplication, QHBoxLayout, QWidget, QSlider,\
    QLabel, QVBoxLayout, QGridLayout
from PySide2.QtCore import Qt
from PySide2.QtGui import QPixmap, QImage, QPalette, QColor, QFont
from matplotlib.backends.backend_qt5agg\
    import FigureCanvasQTAgg as FigureCanvas
from colour import MultiSpectralDistributions, SpectralShape
import matplotlib.pyplot as plt

# import my libraries
import plot_utility as pu
from spectrum import CIE1931_CMFS, DisplaySpectrum, create_display_sd

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


class DisplaySpectrumDataControl():
    def __init__(
            self,
            r_mean_slider, g_mean_slider, b_mean_slider,
            r_dist_slider, g_dist_slider, b_dist_slider):
        self.r_mean_slider = r_mean_slider
        self.g_mean_slider = g_mean_slider
        self.b_mean_slider = b_mean_slider
        self.r_dist_slider = r_dist_slider
        self.g_dist_slider = g_dist_slider
        self.b_dist_slider = b_dist_slider
        msd = self.create_display_spectrum_from_slider()
        self.ds = DisplaySpectrum(msd=msd)

    def create_display_spectrum_from_slider(self):
        r_mu = self.r_mean_slider.get_value()
        g_mu = self.g_mean_slider.get_value()
        b_mu = self.b_mean_slider.get_value()
        r_sigma = self.r_dist_slider.get_value()
        g_sigma = self.r_dist_slider.get_value()
        b_sigma = self.r_dist_slider.get_value()

        ds = create_display_sd(
            r_mu=r_mu, g_mu=g_mu, b_mu=b_mu,
            r_sigma=r_sigma, g_sigma=g_sigma, b_sigma=b_sigma)

        return ds

    def update_display_spectrum(self):
        msd = self.create_display_spectrum_from_slider()
        self.ds.update_msd(msd=msd)


class EventControl():
    def __init__(self) -> None:
        pass

    def set_display_sd_slider_event(
            self, dsd_crtl: DisplaySpectrumDataControl,
            display_sd_canvas,
            r_mean_label, g_mean_label, b_mean_label,
            r_dist_label, g_dist_label, b_dist_label,
            r_mean_slider, g_mean_slider, b_mean_slider,
            r_dist_slider, g_dist_slider, b_dist_slider):

        self.dsd_ctrl = dsd_crtl
        self.display_sd_canvas = display_sd_canvas

        self.r_mean_label = r_mean_label
        self.g_mean_label = g_mean_label
        self.b_mean_label = b_mean_label
        self.r_dist_label = r_dist_label
        self.g_dist_label = g_dist_label
        self.b_dist_label = b_dist_label
        self.r_mean_slider = r_mean_slider
        self.g_mean_slider = g_mean_slider
        self.b_mean_slider = b_mean_slider
        self.r_dist_slider = r_dist_slider
        self.g_dist_slider = g_dist_slider
        self.b_dist_slider = b_dist_slider

        self.r_mean_slider.set_slot(self.display_sd_slider_event)
        self.g_mean_slider.set_slot(self.display_sd_slider_event)
        self.b_mean_slider.set_slot(self.display_sd_slider_event)
        self.r_dist_slider.set_slot(self.display_sd_slider_event)
        self.g_dist_slider.set_slot(self.display_sd_slider_event)
        self.b_dist_slider.set_slot(self.display_sd_slider_event)

    def display_sd_slider_event(self):
        r_mean_value = self.r_mean_slider.get_value()
        g_mean_value = self.g_mean_slider.get_value()
        b_mean_value = self.b_mean_slider.get_value()
        r_dist_value = self.r_dist_slider.get_value()
        g_dist_value = self.g_dist_slider.get_value()
        b_dist_value = self.b_dist_slider.get_value()

        self.r_mean_label.set_label(r_mean_value)
        self.g_mean_label.set_label(g_mean_value)
        self.b_mean_label.set_label(b_mean_value)
        self.r_dist_label.set_label(r_dist_value)
        self.g_dist_label.set_label(g_dist_value)
        self.b_dist_label.set_label(b_dist_value)

        self.dsd_ctrl.update_display_spectrum()


class WindowColorControl():
    def __init__(self, parent) -> None:
        self.parent = parent

    def set_bg_color(self, color=[0.15, 0.15, 0.15]):
        palette = self.parent.palette()
        color_value = [int(x * 255) for x in color]
        palette.setColor(QPalette.Window, QColor(*color_value))
        self.parent.setPalette(palette)
        self.parent.setAutoFillBackground(True)


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


class DisplaySpectrumPlot():
    def __init__(
            self, dsd: DisplaySpectrum,
            figsize=(10, 6)):
        super().__init__()
        spectral_shape = SpectralShape(380, 780, 1)
        self.figsize = figsize
        self.cmfs = CIE1931_CMFS.trim(spectral_shape)
        self.dsd = dsd
        self.init_plot()

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
        sd_wavelength = self.dsd.msd.domain
        sd_white = self.dsd.msd.values[..., 3]
        self.display_sd_line, = ax1.plot(
            sd_wavelength, sd_white, '-',
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


class LayoutControl():
    def __init__(self, parent) -> None:
        # self.base_layout = QHBoxLayout(parent)
        self.base_layout = QGridLayout()
        parent.setLayout(self.base_layout)

    def set_mpl_layout(
            self,
            canvas,
            # chromaticity_diagram_canvas,
            r_mean_label, g_mean_label, b_mean_label,
            r_dist_label, g_dist_label, b_dist_label,
            r_mean_slider, g_mean_slider, b_mean_slider,
            r_dist_slider, g_dist_slider, b_dist_slider,
            # color_temp_slider, color_temp_label,
            # cie1931_xyY_label, cie2012_xyY_label,
            # cie1931_dx_label, cie1931_dy_label
            ):
        r_mu_layout = QHBoxLayout()
        r_sigma_layout = QHBoxLayout()
        g_mu_layout = QHBoxLayout()
        g_sigma_layout = QHBoxLayout()
        b_mu_layout = QHBoxLayout()
        b_sigma_layout = QHBoxLayout()

        # color_temp_layout = QHBoxLayout()

        r_mu_layout.addWidget(r_mean_label.get_widget())
        r_mu_layout.addWidget(r_mean_slider.get_widget())
        r_sigma_layout.addWidget(r_dist_label.get_widget())
        r_sigma_layout.addWidget(r_dist_slider.get_widget())

        g_mu_layout.addWidget(g_mean_label.get_widget())
        g_mu_layout.addWidget(g_mean_slider.get_widget())
        g_sigma_layout.addWidget(g_dist_label.get_widget())
        g_sigma_layout.addWidget(g_dist_slider.get_widget())

        b_mu_layout.addWidget(b_mean_label.get_widget())
        b_mu_layout.addWidget(b_mean_slider.get_widget())
        b_sigma_layout.addWidget(b_dist_label.get_widget())
        b_sigma_layout.addWidget(b_dist_slider.get_widget())

        # color_temp_layout.addWidget(color_temp_label.get_widget())
        # color_temp_layout.addWidget(color_temp_slider.get_widget())

        mpl_layout = QVBoxLayout()
        mpl_layout.addWidget(canvas.get_widget())
        mpl_layout.addLayout(r_mu_layout)
        mpl_layout.addLayout(r_sigma_layout)
        mpl_layout.addLayout(g_mu_layout)
        mpl_layout.addLayout(g_sigma_layout)
        mpl_layout.addLayout(b_mu_layout)
        mpl_layout.addLayout(b_sigma_layout)

        # mpl_layout.addLayout(color_temp_layout)

        # cie_xyY_layout = QHBoxLayout()
        # cie_xyY_layout.addWidget(cie1931_xyY_label.get_widget())
        # cie_xyY_layout.addWidget(cie2012_xyY_label.get_widget())
        # cie_dxy_layout = QHBoxLayout()
        # cie_dxy_layout.addWidget(cie1931_dx_label.get_widget())
        # cie_dxy_layout.addWidget(cie1931_dy_label.get_widget())

        # chroma_diagram_layout = QVBoxLayout()
        # chroma_diagram_layout.addWidget(
        #     chromaticity_diagram_canvas.get_widget())
        # chroma_diagram_layout.addLayout(cie_xyY_layout)
        # chroma_diagram_layout.addLayout(cie_dxy_layout)

        self.base_layout.addLayout(mpl_layout, 0, 0)
        # self.base_layout.addLayout(chroma_diagram_layout, 0, 1)


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1280, 720)

        # background color
        window_color = WindowColorControl(parent=self)
        window_color.set_bg_color(color=[0.8, 0.8, 0.8])

        # layout
        layout = LayoutControl(self)

        # object for widget
        r_mean_slider = TyBasicSlider(
            int_float_rate=1, default=700, min_val=525, max_val=750)
        g_mean_slider = TyBasicSlider(
            int_float_rate=1, default=546, min_val=430, max_val=630)
        b_mean_slider = TyBasicSlider(
            int_float_rate=1, default=435, min_val=360, max_val=560)
        r_dist_slider = TyBasicSlider(
            int_float_rate=1, default=25, min_val=1, max_val=150)
        g_dist_slider = TyBasicSlider(
            int_float_rate=1, default=25, min_val=1, max_val=150)
        b_dist_slider = TyBasicSlider(
            int_float_rate=1, default=25, min_val=1, max_val=150)

        r_mean_label = TyBasicLabel(
            default=r_mean_slider.get_default(),
            prefix="R_mu:", suffix="[nm]")
        g_mean_label = TyBasicLabel(
            default=g_mean_slider.get_default(),
            prefix="G_mu:", suffix="[nm]")
        b_mean_label = TyBasicLabel(
            default=b_mean_slider.get_default(),
            prefix="B_mu:", suffix="[nm]")
        r_dist_label = TyBasicLabel(
            default=r_dist_slider.get_default(), prefix=" R_sd:", suffix="")
        g_dist_label = TyBasicLabel(
            default=g_dist_slider.get_default(), prefix=" G_sd:", suffix="")
        b_dist_label = TyBasicLabel(
            default=b_dist_slider.get_default(), prefix=" B_sd:", suffix="")

        # display spectral distribution
        dsd_ctrl = DisplaySpectrumDataControl(
            r_mean_slider=r_mean_slider, g_mean_slider=g_mean_slider,
            b_mean_slider=b_mean_slider,
            r_dist_slider=r_dist_slider, g_dist_slider=g_dist_slider,
            b_dist_slider=b_dist_slider)

        display_sd_plot = DisplaySpectrumPlot(
            figsize=(10, 6), dsd=dsd_ctrl.ds)

        layout.set_mpl_layout(
            canvas=display_sd_plot,
            r_mean_label=r_mean_label, g_mean_label=g_mean_label,
            b_mean_label=b_mean_label, r_dist_label=r_dist_label,
            g_dist_label=g_dist_label, b_dist_label=b_dist_label,
            r_mean_slider=r_mean_slider, g_mean_slider=g_mean_slider,
            b_mean_slider=b_mean_slider, r_dist_slider=r_dist_slider,
            g_dist_slider=g_dist_slider, b_dist_slider=b_dist_slider
        )

        # set slot
        self.event_control = EventControl()
        # self.event_control.set_white_slider_event(
        #     white_slider=white_slider, white_label=white_label,
        #     spectrum_plot=spectrum_plot, patch_img=color_checkr_img)
        self.event_control.set_display_sd_slider_event(
            dsd_crtl=dsd_ctrl, display_sd_canvas=display_sd_plot,
            r_mean_label=r_mean_label, g_mean_label=g_mean_label,
            b_mean_label=b_mean_label, r_dist_label=r_dist_label,
            g_dist_label=g_dist_label, b_dist_label=b_dist_label,
            r_mean_slider=r_mean_slider, g_mean_slider=g_mean_slider,
            b_mean_slider=b_mean_slider, r_dist_slider=r_dist_slider,
            g_dist_slider=g_dist_slider, b_dist_slider=b_dist_slider)


def main_func():
    app = QApplication([])
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
