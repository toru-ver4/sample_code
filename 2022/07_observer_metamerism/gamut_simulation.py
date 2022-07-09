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
from colour import SpectralShape
import matplotlib.pyplot as plt

# import my libraries
import color_space as cs
import plot_utility as pu
from spectrum import CIE1931_CMFS, DisplaySpectrum, create_display_sd,\
    START_WAVELENGTH, STOP_WAVELENGTH, WAVELENGTH_STEP

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


class SpdPlotObjects():
    """
    Manage GUI objects related to SPD plot.
    """
    def __init__(self):
        self.create_objects()

    def create_objects(self):
        # object for widget
        self.r_mean_slider = TyBasicSlider(
            int_float_rate=1, default=600, min_val=555, max_val=700)
        self.g_mean_slider = TyBasicSlider(
            int_float_rate=1, default=546, min_val=450, max_val=650)
        self.b_mean_slider = TyBasicSlider(
            int_float_rate=1, default=435, min_val=380, max_val=550)
        self.r_dist_slider = TyBasicSlider(
            int_float_rate=1, default=25, min_val=1, max_val=150)
        self.g_dist_slider = TyBasicSlider(
            int_float_rate=1, default=25, min_val=1, max_val=150)
        self.b_dist_slider = TyBasicSlider(
            int_float_rate=1, default=25, min_val=1, max_val=150)

        self.r_mean_label = TyBasicLabel(
            default=self.r_mean_slider.get_default(),
            prefix="R_mu:", suffix="[nm]")
        self.g_mean_label = TyBasicLabel(
            default=self.g_mean_slider.get_default(),
            prefix="G_mu:", suffix="[nm]")
        self.b_mean_label = TyBasicLabel(
            default=self.b_mean_slider.get_default(),
            prefix="B_mu:", suffix="[nm]")
        self.r_dist_label = TyBasicLabel(
            default=self.r_dist_slider.get_default(), prefix=" R_sd:")
        self.g_dist_label = TyBasicLabel(
            default=self.g_dist_slider.get_default(), prefix=" G_sd:")
        self.b_dist_label = TyBasicLabel(
            default=self.b_dist_slider.get_default(), prefix=" B_sd:")

    def create_spectrum_plot_obj(self, dsd_ctrl):
        self.display_sd_plot = DisplaySpectrumPlot(
            figsize=(10, 6), dsd=dsd_ctrl.ds)

    def get_spd_gen_params(self):
        return\
            self.r_mean_slider.get_value(), self.r_dist_slider.get_value(),\
            self.g_mean_slider.get_value(), self.g_dist_slider.get_value(),\
            self.b_mean_slider.get_value(), self.b_dist_slider.get_value()


class DisplaySpectrumDataControl():
    def __init__(self):
        msd = self.create_display_spectrum_from_slider(
            r_mu=600, r_sigma=10, g_mu=550, g_sigma=20,
            b_mu=450, b_sigma=30)
        self.ds = DisplaySpectrum(msd=msd)

    def create_display_spectrum_from_slider(
            self, r_mu, r_sigma, g_mu, g_sigma, b_mu, b_sigma):

        ds = create_display_sd(
            r_mu=r_mu, g_mu=g_mu, b_mu=b_mu,
            r_sigma=r_sigma, g_sigma=g_sigma, b_sigma=b_sigma)

        return ds

    def update_display_spectrum(
            self, r_mu, r_sigma, g_mu, g_sigma, b_mu, b_sigma):
        msd = self.create_display_spectrum_from_slider(
            r_mu=r_mu, g_mu=g_mu, b_mu=b_mu,
            r_sigma=r_sigma, g_sigma=g_sigma, b_sigma=b_sigma)
        self.ds.update_msd(msd=msd)


class ChromaticityDiagramPlotObjects():
    def __init__(
            self, dsd_ctrl: DisplaySpectrumDataControl):
        self.create_objects(dsd_ctrl=dsd_ctrl)

    def create_objects(self, dsd_ctrl):
        self.mpl_obj = ChromaticityDiagramPlot(dsd_ctrl=dsd_ctrl)

    def get_canvas(self):
        return self.mpl_obj.get_widget()


class ChromaticityDiagramPlot():
    def __init__(
            self, dsd_ctrl: DisplaySpectrumDataControl):
        self.ds = dsd_ctrl.ds
        rate = 1.0
        xmin = -0.1
        xmax = 0.8
        ymin = -0.1
        ymax = 1.0
        st_wl = 380
        ed_wl = 780
        wl_step = 1
        self.cmf_xy = pu.calc_horseshoe_chromaticity(
            st_wl=st_wl, ed_wl=ed_wl, wl_step=wl_step)
        self.cmf_xy_norm = pu.calc_normal_pos(
            xy=self.cmf_xy, normal_len=0.05, angle_degree=90)
        self.wl_list = np.arange(st_wl, ed_wl + 1, wl_step)

        self.xy_image = pu.get_chromaticity_image(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cmf_xy=self.cmf_xy)
        self.plot_chromaticity_diagram_init(
            rate=rate, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    def plot_chromaticity_diagram_init(
            self, rate=1.0, xmin=-0.1, xmax=0.8, ymin=-0.1, ymax=1.0):
        plot_wl_list = [
            410, 450, 470, 480, 485, 490, 495,
            500, 505, 510, 520, 530, 540, 550, 560, 570, 580, 590,
            600, 620, 690]
        fig, ax1 = pu.plot_1_graph(
            fontsize=20 * rate,
            figsize=((xmax - xmin) * 10 * rate,
                     (ymax - ymin) * 10 * rate),
            graph_title="CIE1931 Chromaticity Diagram",
            graph_title_size=None,
            xlabel=None, ylabel=None,
            axis_label_size=None,
            legend_size=14 * rate,
            xlim=(xmin, xmax),
            ylim=(ymin, ymax),
            xtick=[x * 0.1 + xmin for x in
                   range(int((xmax - xmin)/0.1) + 1)],
            ytick=[x * 0.1 + ymin for x in
                   range(int((ymax - ymin)/0.1) + 1)],
            xtick_size=17 * rate,
            ytick_size=17 * rate,
            linewidth=4 * rate,
            minor_xtick_num=2,
            minor_ytick_num=2)
        ax1.plot(
            self.cmf_xy[..., 0], self.cmf_xy[..., 1], '-k', lw=2*rate,
            label=None)
        for idx, wl in enumerate(self.wl_list):
            if wl not in plot_wl_list:
                continue
            pu.draw_wl_annotation(
                ax1=ax1, wl=wl, rate=rate,
                st_pos=[self.cmf_xy_norm[idx, 0], self.cmf_xy_norm[idx, 1]],
                ed_pos=[self.cmf_xy[idx, 0], self.cmf_xy[idx, 1]])
        bt709_gamut = pu.get_primaries(name=cs.BT709)
        ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1],
                 c=pu.RED, label="BT.709", lw=2.75*rate)
        bt2020_gamut = pu.get_primaries(name=cs.BT2020)
        ax1.plot(bt2020_gamut[:, 0], bt2020_gamut[:, 1],
                 c=pu.GREEN, label="BT.2020", lw=2.75*rate)
        dci_p3_gamut = pu.get_primaries(name=cs.P3_D65)
        ax1.plot(dci_p3_gamut[:, 0], dci_p3_gamut[:, 1],
                 c=pu.BLUE, label="DCI-P3", lw=2.75*rate)
        primaries = self.ds.primaries
        white = self.ds.white
        self.primaries_line, = ax1.plot(
            primaries[:, 0], primaries[:, 1],
            c='k', label="Display Gamut", lw=2.75*rate)
        ax1.plot(
            [0.3127], [0.3290], 'o', label='D65', ms=14*rate,
            color=[0.7, 0.7, 0.7], alpha=1.0)
        ax1.plot(
            white[0], white[1], 'x', label='Display White',
            ms=12*rate, mew=2*rate, color='k')
        ax1.imshow(self.xy_image, extent=(xmin, xmax, ymin, ymax), alpha=0.5)
        plt.legend(loc='upper right')
        self.canvas = FigureCanvas(fig)

    def update_plot(self):
        primaries = self.ds.primaries
        self.primaries_line.set_data(primaries[:, 0], primaries[:, 1])
        self.canvas.draw()
        # self.primaries_line.figure.canvas.draw()

    def get_widget(self):
        return self.canvas


class EventControl():
    def __init__(self) -> None:
        pass

    def set_display_sd_slider_event(
            self, dsd_crtl: DisplaySpectrumDataControl,
            spd_objects: SpdPlotObjects,
            chromaticity_objects: ChromaticityDiagramPlotObjects):

        self.dsd_ctrl = dsd_crtl
        self.spd_objects = spd_objects
        self.display_sd_canvas = self.spd_objects.display_sd_plot
        self.chromaticity_diagram_obj = chromaticity_objects.mpl_obj

        self.spd_objects.r_mean_slider.set_slot(self.display_sd_slider_event)
        self.spd_objects.g_mean_slider.set_slot(self.display_sd_slider_event)
        self.spd_objects.b_mean_slider.set_slot(self.display_sd_slider_event)
        self.spd_objects.r_dist_slider.set_slot(self.display_sd_slider_event)
        self.spd_objects.g_dist_slider.set_slot(self.display_sd_slider_event)
        self.spd_objects.b_dist_slider.set_slot(self.display_sd_slider_event)

    def display_sd_slider_event(self):
        r_mu, r_sigma, g_mu, g_sigma, b_mu, b_sigma =\
            self.spd_objects.get_spd_gen_params()

        self.spd_objects.r_mean_label.set_label(r_mu)
        self.spd_objects.g_mean_label.set_label(g_mu)
        self.spd_objects.b_mean_label.set_label(b_mu)
        self.spd_objects.r_dist_label.set_label(r_sigma)
        self.spd_objects.g_dist_label.set_label(g_sigma)
        self.spd_objects.b_dist_label.set_label(b_sigma)

        self.dsd_ctrl.update_display_spectrum(
            r_mu=r_mu, r_sigma=r_sigma,
            g_mu=g_mu, g_sigma=g_sigma,
            b_mu=b_mu, b_sigma=b_sigma)

        self.display_sd_canvas.update_plot()
        # self.chromaticity_diagram_obj.update_plot()


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
        spectral_shape = SpectralShape(
            START_WAVELENGTH, STOP_WAVELENGTH, WAVELENGTH_STEP)
        self.figsize = figsize
        self.cmfs = CIE1931_CMFS.trim(spectral_shape)
        self.dsd = dsd
        self.init_plot()

    def init_plot(self):
        self.fig, self.ax1 = pu.plot_1_graph(
            fontsize=14,
            figsize=self.figsize,
            graph_title="Spectral power distribution",
            graph_title_size=None,
            xlabel="Wavelength [nm]", ylabel="Relative power",
            axis_label_size=None,
            legend_size=12,
            xlim=[360, 730],
            ylim=[-0.05, 2.05],
            xtick=None,
            ytick=None,
            xtick_size=None, ytick_size=None,
            linewidth=3,
            return_figure=True)
        sd_wavelength = self.dsd.msd.domain
        self.display_sd_line_w, = self.ax1.plot(
            sd_wavelength, self.dsd.msd.values[..., 3], '-',
            color=(0.1, 0.1, 0.1), label="Display (W=R+G+B)", lw=5)
        self.display_sd_line_r, = self.ax1.plot(
            sd_wavelength, self.dsd.msd.values[..., 0], '-',
            color=pu.RED, label="Display (R)", lw=1.5)
        self.display_sd_line_g, = self.ax1.plot(
            sd_wavelength, self.dsd.msd.values[..., 1], '-',
            color=pu.GREEN, label="Display (G)", lw=1.5)
        self.display_sd_line_b, = self.ax1.plot(
            sd_wavelength, self.dsd.msd.values[..., 2], '-',
            color=pu.SKY, label="Display (B)", lw=1.5)
        self.ax1.plot(
            self.cmfs.wavelengths, self.cmfs.values[..., 0], '--',
            color=pu.RED, label="CIE 1931 2 CMF(R)", lw=1)
        self.ax1.plot(
            self.cmfs.wavelengths, self.cmfs.values[..., 1], '--',
            color=pu.GREEN, label="CIE 1931 2 CMF(G)", lw=1)
        self.ax1.plot(
            self.cmfs.wavelengths, self.cmfs.values[..., 2], '--',
            color=pu.BLUE, label="CIE 1931 2 CMF(B)", lw=1)

        plt.legend(loc='upper right')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax1.bbox)

    def update_plot(self):
        sd_wavelength = self.dsd.msd.domain
        update_list = [
            [self.display_sd_line_r, self.dsd.msd.values[..., 0]],
            [self.display_sd_line_g, self.dsd.msd.values[..., 1]],
            [self.display_sd_line_b, self.dsd.msd.values[..., 2]],
            [self.display_sd_line_w, self.dsd.msd.values[..., 3]]]
        self.fig.canvas.restore_region(self.background)
        for update_info in update_list:
            display_sd_line_obj = update_info[0]
            display_sd_data = update_info[1]

            display_sd_line_obj.set_data(sd_wavelength, display_sd_data)
            self.ax1.draw_artist(display_sd_line_obj)
        self.fig.canvas.blit(self.ax1.bbox)

            # display_sd_line_obj.figure.canvas.draw()

    def get_widget(self):
        return self.canvas


class LayoutControl():
    def __init__(self, parent) -> None:
        # self.base_layout = QHBoxLayout(parent)
        self.base_layout = QGridLayout()
        parent.setLayout(self.base_layout)

    def set_mpl_layout(
            self, spd_objcts: SpdPlotObjects,
            chromaticity_objects: ChromaticityDiagramPlotObjects
            # canvas,
            # chromaticity_diagram_canvas,
            # r_mean_label, g_mean_label, b_mean_label,
            # r_dist_label, g_dist_label, b_dist_label,
            # r_mean_slider, g_mean_slider, b_mean_slider,
            # r_dist_slider, g_dist_slider, b_dist_slider,
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

        r_mu_layout.addWidget(spd_objcts.r_mean_label.get_widget())
        r_mu_layout.addWidget(spd_objcts.r_mean_slider.get_widget())
        r_sigma_layout.addWidget(spd_objcts.r_dist_label.get_widget())
        r_sigma_layout.addWidget(spd_objcts.r_dist_slider.get_widget())

        g_mu_layout.addWidget(spd_objcts.g_mean_label.get_widget())
        g_mu_layout.addWidget(spd_objcts.g_mean_slider.get_widget())
        g_sigma_layout.addWidget(spd_objcts.g_dist_label.get_widget())
        g_sigma_layout.addWidget(spd_objcts.g_dist_slider.get_widget())

        b_mu_layout.addWidget(spd_objcts.b_mean_label.get_widget())
        b_mu_layout.addWidget(spd_objcts.b_mean_slider.get_widget())
        b_sigma_layout.addWidget(spd_objcts.b_dist_label.get_widget())
        b_sigma_layout.addWidget(spd_objcts.b_dist_slider.get_widget())

        # color_temp_layout.addWidget(color_temp_label.get_widget())
        # color_temp_layout.addWidget(color_temp_slider.get_widget())

        mpl_layout = QVBoxLayout()
        canvas = spd_objcts.display_sd_plot
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

        chroma_diagram_layout = QVBoxLayout()
        chroma_diagram_layout.addWidget(
            chromaticity_objects.get_canvas())
        # chroma_diagram_layout.addLayout(cie_xyY_layout)
        # chroma_diagram_layout.addLayout(cie_dxy_layout)

        self.base_layout.addLayout(mpl_layout, 0, 0)
        self.base_layout.addLayout(chroma_diagram_layout, 0, 1)


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1920, 1080)

        # background color
        window_color = WindowColorControl(parent=self)
        window_color.set_bg_color(color=[0.8, 0.8, 0.8])

        spd_objects = SpdPlotObjects()
        dsd_ctrl = DisplaySpectrumDataControl()
        spd_objects.create_spectrum_plot_obj(dsd_ctrl=dsd_ctrl)

        chromaticity_diagram_objects = ChromaticityDiagramPlotObjects(
            dsd_ctrl=dsd_ctrl)

        # layout
        layout = LayoutControl(self)
        layout.set_mpl_layout(
            spd_objcts=spd_objects,
            chromaticity_objects=chromaticity_diagram_objects)

        # set slot
        self.event_control = EventControl()
        # self.event_control.set_white_slider_event(
        #     white_slider=white_slider, white_label=white_label,
        #     spectrum_plot=spectrum_plot, patch_img=color_checkr_img)
        self.event_control.set_display_sd_slider_event(
            dsd_crtl=dsd_ctrl, spd_objects=spd_objects,
            chromaticity_objects=chromaticity_diagram_objects)


def main_func():
    app = QApplication([])
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
