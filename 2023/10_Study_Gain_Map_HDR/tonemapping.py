# -*- coding: utf-8 -*-
"""
improve the 3dlut.
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import scipy.optimize
from sympy import symbols, solve
from sympy.utilities.lambdify import lambdify

# import my libraries
import transfer_functions as tf
import plot_utility as pu
from scipy import interpolate

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_quadratic_bezier_function(control_points):
    """
    Create an optimized function that calculates y-coordinates on a quadratic Bezier curve for given x-coordinates.

    Parameters:
    control_points (list of tuples): Control points for the Bezier curve [(x0, y0), (x1, y1), (x2, y2)]

    Returns:
    function: A function that takes a NumPy array of x-coordinates and returns the corresponding y-coordinates
    """
    # Extract control points
    (x0, y0), (x1, y1), (x2, y2) = control_points

    # Define the Bezier curve equation
    def bezier_eq(t, x):
        return (1 - t)**2 * x0 + 2 * (1 - t) * t * x1 + t**2 * x2 - x

    def bezier_function(x_values):
        # Vectorized root finding for each x value
        t_solutions = np.array([scipy.optimize.newton(lambda t: bezier_eq(t, x), 0.5) for x in x_values])

        # Vectorized calculation of y values
        y_values = (1 - t_solutions)**2 * y0 + 2 * (1 - t_solutions) * t_solutions * y1 + t_solutions**2 * y2

        return y_values

    return bezier_function


def get_top_side_bezier(params=[[0.6, 0.7], [0.8, 1.0], [1.0, 1.0]]):
    """
    Note
    ----
    An example of ```top_param``` is bellow.
        {'x0': 0.5, 'y0': 0.5,
         'x1': 0.7, 'y1': 0.7,
         'x2': 1.0, 'y2': 0.7}
    """
    a_val = params[0][0]
    b_val = params[1][0]
    c_val = params[2][0]
    p_val = params[0][1]
    q_val = params[1][1]
    r_val = params[2][1]

    a, b, c, t, x = symbols('a, b, c, t, x')
    f = (1 - t)**2 * a + 2 * (1 - t) * t * b + t**2 * c - x

    # x について解く
    # ----------------------
    t = solve(f, t)[1]
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


def ty_tonemap_v2(
        x, ks_hdr_luminance=203, ks_sdr_luminance=100,
        ke_hdr_luminance=1000, ke_sdr_luminance=203):
    """
    YouTube の HDR to SDR のトーンマップを模倣してみる。
    中間階調までは直線で、高階調部だけ2次ベジェ曲線で丸める。

    直線の数式は $y = 0.74x + 0.01175$。
    y軸の最大値は 0.508078421517 (100nits)。
    この時の x は $x = (0.508078421517 - 0.01175) / 0.74$ より
    0.6707140831310812 。ちなみに、リニアなら 473.5 nits。

    ks は knee start の略。ke は knee end.

    Parameters
    ----------
    x : ndarray
        input data. range is 0.0-1.0.
    ks_hdr_luminance : float
        knee start luminance in HDR. the unit is cd/m2.
    ks_sdr_luminance : float
        knee start luminance in SDR. the unit is cd/m2.
    ke_hdr_luminance : float
        knee end luminance in HDR. the unit is cd/m2.
    ke_sdr_luminance : float
        knee end luminance in SDR. the unit is cd/m2.
    """
    ks_x = tf.oetf_from_luminance(ks_hdr_luminance, tf.ST2084)
    ks_y = tf.oetf_from_luminance(ks_sdr_luminance, tf.ST2084)
    ke_x = tf.oetf_from_luminance(ke_hdr_luminance, tf.ST2084)
    ke_y = tf.oetf_from_luminance(ke_sdr_luminance, tf.ST2084)
    slope = ks_y / ks_x
    # mid_x = ke_y / slope
    mid_x = (ke_x + ks_x) / 2.0
    mid_y = ke_y
    print(slope)
    params = [[ks_x, ks_y], [mid_x, mid_y], [ke_x, ke_y]]
    # bezie = get_top_side_bezier(params=params)
    bezie = create_quadratic_bezier_function(
        control_points=[(ks_x, ks_y), (mid_x, mid_y), (ke_x, ke_y)])
    y = np.select(
        (x < ks_x, x <= ke_x, x > ke_x),
        (slope * x, bezie(x), ke_y))
    params = [[0, 0], [ks_x, ks_y], [mid_x, mid_y], [ke_x, ke_y], [1.0, ke_y]]
    print(params)

    return y


def create_ty_tonemap_v2_func():
    x = np.linspace(0, 1, 1024)
    y = ty_tonemap_v2(x=x)
    func = interpolate.interp1d(x, y)
    return func


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    x = np.linspace(0, 1, 1024)
    y = ty_tonemap_v2(x)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Title",
        graph_title_size=None,
        xlabel="X Axis Label",
        ylabel="Y Axis Label",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=[-0.05, 1.05],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y, label=None)
    pu.show_and_save(fig=fig, legend_loc='upper left', show=True)
