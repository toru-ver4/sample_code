#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
## 概要

* Color Volume 変換関連のライブラリ
* とりあえず Tone Mapping 関数を置く
* 将来的には色域変換とかも入れたい

"""

import os
import numpy as np
from sympy import init_printing, pprint
from sympy import symbols, solve
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
import plot_utility as pu
import colour


def get_top_side_bezier(kind="top", **kwargs):

    a_val = kwargs['x0']
    b_val = kwargs['x1']
    c_val = kwargs['x2']
    p_val = kwargs['y0']
    q_val = kwargs['y1']
    r_val = kwargs['y2']

    a, b, c, t, x = symbols('a, b, c, t, x')
    f = (1 - t)**2 * a + 2 * (1 - t) * t * b + t**2 * c - x

    # x について解く
    # ----------------------
    if kind == 'top':
        t = solve(f, t)[1]
    elif kind == 'bottom':
        t = solve(f, t)[0]
    else:
        raise ValueError("kind parameter is invalid.")
    t = t.subs({a: a_val, b: b_val, c: c_val})

    # y と t(ここでは u と置いた) の関係式を記述
    # -------------------------------------------
    p, q, r, u, y = symbols('p, q, r, u, y')
    y = (1 - u)**2 * p + 2 * (1 - u) * u * q + u**2 * r

    # パラメータ u と事前に求めた t で置き換える
    # -------------------------------------------
    y = y.subs({p: p_val, q: q_val, r: r_val, u: t})

    func = lambdify(x, y, 'numpy')

    return func


def tonemap_2dim_bezier(x, top_param, bottom_param, plot=False):
    """
    2次ベジェ曲線でトーンマップする

    Parameters
    ----------
    x : array_like
        target data.
    top_param : dictionary
        tonemapping parameters for top area.
    bottom_param : dictionary
        tonemapping parameters for bottom area.
    plot : boolean
        whether plotting or not.

    Note
    ----
    An example of ```top_param``` is bellow.
        {'x0': 0.5, 'y0': 0.5,
         'x1': 0.7, 'y1': 0.7,
         'x2': 1.0, 'y2': 0.7}
    An example of ```bottom_param``` is bellow.
        {'x0': 0.0, 'y0': 0.1,
        'x1': 0.1, 'y1': 0.1,
        'x2': 0.3, 'y2': 0.3}

    """

    temp = tonemap_2dim_bezier_bottom(x, plot=False, **bottom_param)
    y = tonemap_2dim_bezier_top(temp, plot=False, **top_param)

    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="Title",
                              graph_title_size=None,
                              xlabel="X Axis Label", ylabel="Y Axis Label",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=None,
                              ylim=None,
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=3)
        ax1.plot(x, y, label='bezier')
        plt.legend(loc='upper left')
        plt.show()

    return y


def tonemap_2dim_bezier_top(x, plot=False, **kwargs):
    """
    2次ベジェ曲線でトーンマップする。主に上の方。
    """

    y = np.zeros_like(x)
    low_idx = (x <= kwargs['x0'])
    middle_idx = (kwargs['x0'] < x) & (x <= kwargs['x2'])
    high_idx = (kwargs['x2'] < x)

    func = get_top_side_bezier(kind='top', **kwargs)

    y[low_idx] = x[low_idx].copy()
    y[middle_idx] = func(x[middle_idx].copy())
    y[high_idx] = kwargs['y2']

    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="Title",
                              graph_title_size=None,
                              xlabel="X Axis Label", ylabel="Y Axis Label",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=None,
                              ylim=None,
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=3)
        ax1.plot(x, y, label='bezier')
        plt.legend(loc='upper left')
        plt.show()

    return y


def tonemap_2dim_bezier_bottom(x, plot=False, **kwargs):
    """
    2次ベジェ曲線でトーンマップする。主に下の方。

    Example
    -------
    x = np.linspace(0, 1, 1024)
    x2 = colour.models.eotf_ST2084(x) / 10000
    bottom_param = {'x0': 0.00, 'y0': 0.001,
                    'x1': 0.001, 'y1': 0.001,
                    'x2': 0.005, 'y2': 0.005}
    top_param = {'x0': 0.02, 'y0': 0.02,
                 'x1': 0.03, 'y1': 0.03,
                 'x2': 0.12, 'y2': 0.03}
    y = tonemap_2dim_bezier(x2, bottom_param=bottom_param,
                            top_param=top_param, plot=False)
    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title="Title",
                          graph_title_size=None,
                          xlabel="Video Level",
                          ylabel="Y Axis Label",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=None,
                          ylim=[0, 500],
                          xtick=[0, 256, 512, 768, 1024],
                          ytick=None,
                          xtick_size=None, ytick_size=None,
                          linewidth=3)
    ax1.plot(x * 1023, x2 * 10000, label='st2084')
    ax1.plot(x * 1023, y * 10000, label='bezier')
    plt.legend(loc='upper left')
    plt.show()
    """

    y = np.zeros_like(x)
    low_idx = (x <= kwargs['x0'])
    middle_idx = (kwargs['x0'] < x) & (x <= kwargs['x2'])
    high_idx = (kwargs['x2'] < x)

    func = get_top_side_bezier(kind='top', **kwargs)

    y[low_idx] = kwargs['y0']
    y[middle_idx] = func(x[middle_idx].copy())
    y[high_idx] = x[high_idx].copy()

    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="Title",
                              graph_title_size=None,
                              xlabel="X Axis Label", ylabel="Y Axis Label",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=None,
                              ylim=None,
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=3)
        ax1.plot(x, y, label='bezier')
        plt.legend(loc='upper left')
        plt.show()

    return y


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
