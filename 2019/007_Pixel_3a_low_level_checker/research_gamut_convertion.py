#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gamut変換の特性を調べる。
"""

# 外部ライブラリのインポート
import os
import numpy as np
import matplotlib.pyplot as plt

# 自作ライブラリのインポート
import test_pattern_generator2 as tpg
import color_space as cs
import transfer_functions as tf
from CalcParameters import CalcParameters
import plot_utility as pu

UNIVERSAL_COLOR_LIST = ["#F6AA00", "#FFF100", "#03AF7A",
                        "#005AFF", "#4DC4FF", "#804000"]

base_param = {
    'revision': 1,
    'inner_sample_num': 4,
    'outer_sample_num': 6,
    'hue_devide_num': 4,
    'img_width': 1920,
    'img_height': 1080,
    'pattern_space_rate': 0.71,
    'inner_gamut_name': 'ITR-R BT.709',
    'outer_gamut_name': 'ITU-R BT.2020',
    'inner_primaries': np.array(tpg.get_primaries(cs.BT709)[0]),
    'outer_primaries': np.array(tpg.get_primaries(cs.BT2020)[0]),
    'transfer_function': tf.SRGB,
    'background_luminance': 2,
    'reference_white': 100
}


def plot_chromaticity_diagram(
        base_param, outer_xyY, outer_ref_xyY,
        xmin=0.0, xmax=0.8, ymin=0.0, ymax=0.9):
    xy_image = tpg.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    cmf_xy = tpg._get_cmfs_xy()
    xlim = (min(0, xmin), max(0.8, xmax))
    ylim = (min(0, ymin), max(0.9, ymax))
    figsize_h = 8 * 1.0
    figsize_v = 9 * 1.0
    rate = 1.3
    # gamut の用意
    outer_gamut = base_param['outer_primaries']
    inner_gamut = base_param['inner_primaries']
    outer_name = base_param['outer_gamut_name']
    inner_name = base_param['inner_gamut_name']
    ax1 = pu.plot_1_graph(fontsize=20 * rate,
                          figsize=(figsize_h, figsize_v),
                          graph_title="CIE1931 Chromaticity Diagram",
                          xlabel=None, ylabel=None,
                          legend_size=18 * rate,
                          xlim=xlim, ylim=ylim,
                          xtick=[x * 0.1 + xmin for x in
                                 range(int((xlim[1] - xlim[0])/0.1) + 1)],
                          ytick=[x * 0.1 + ymin for x in
                                 range(int((ylim[1] - ylim[0])/0.1) + 1)],
                          xtick_size=17 * rate,
                          ytick_size=17 * rate,
                          linewidth=4 * rate,
                          minor_xtick_num=2, minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3.5*rate, label=None)
    ax1.plot(inner_gamut[:, 0], inner_gamut[:, 1],
             c=UNIVERSAL_COLOR_LIST[0], label=inner_name, lw=2.75*rate)
    ax1.plot(outer_gamut[:, 0], outer_gamut[:, 1],
             c=UNIVERSAL_COLOR_LIST[3], label=outer_name, lw=2.75*rate)
    ax1.plot(tpg.D65_WHITE[0], tpg.D65_WHITE[1], marker='x', c='k',
             lw=2.75*rate, label='D65', ms=10*rate, mew=2.75*rate)
    ax1.plot(outer_ref_xyY[..., 0], outer_ref_xyY[..., 1], ls='', marker='s',
             c='#808080', ms=8*rate)
    ax1.plot(outer_xyY[..., 0], outer_xyY[..., 1], ls='', marker='o',
             c='#808080', ms=8*rate)
    # annotation
    arrowprops = dict(
        facecolor='#333333', shrink=0.0, headwidth=8, headlength=10,
        width=2)
    # for idx in range(outer_xyY.shape[0]):
    #     ed_pos = (outer_ref_xyY[idx][0], outer_ref_xyY[idx][1])
    #     st_pos = (outer_xyY[idx][0], outer_xyY[idx][1])
    #     ax1.annotate("", xy=ed_pos, xytext=st_pos, xycoords='data',
    #                  textcoords='data', ha='left', va='bottom',
    #                  arrowprops=arrowprops)
    for h_idx in range(outer_xyY.shape[0]):
        for s_idx in range(outer_xyY.shape[1]):
            ed_pos = (outer_ref_xyY[h_idx][s_idx][0],
                      outer_ref_xyY[h_idx][s_idx][1])
            st_pos = (outer_xyY[h_idx][s_idx][0],
                      outer_xyY[h_idx][s_idx][1])
            ax1.annotate("", xy=ed_pos, xytext=st_pos, xycoords='data',
                         textcoords='data', ha='left', va='bottom',
                         arrowprops=arrowprops)

    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax))
    plt.legend(loc='upper right')
    png_file_name = "./are.png"
    plt.savefig(png_file_name, bbox_inches='tight')
    plt.show()


def make_outer_xyY(inner_xyY, outer_xyY):
    """
    outer_xy の index[0] がちょうど inner_gamut の edge に
    なるように、outer_xy にデータを追加する。
    """
    start_index = inner_xyY.shape[1] - 1
    xyY_array = np.append(inner_xyY, outer_xyY, axis=1)
    return xyY_array[:, start_index:, :]


def main_func():
    cp = CalcParameters(base_param)
    draw_param = cp.calc_parameters()
    outer_xyY = make_outer_xyY(
        draw_param['inner_xyY'], draw_param['outer_xyY'])
    outer_ref_xyY = make_outer_xyY(
        draw_param['inner_ref_xyY'], draw_param['outer_ref_xyY'])

    # plot_chromaticity_diagram(base_param, outer_xyY[4], outer_ref_xyY[4])
    plot_chromaticity_diagram(base_param, outer_xyY, outer_ref_xyY)

    print(outer_xyY[4])
    print(outer_ref_xyY[4])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
