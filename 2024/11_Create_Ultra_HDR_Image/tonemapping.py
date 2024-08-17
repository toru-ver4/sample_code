# -*- coding: utf-8 -*-
"""

"""

# import standard libraries

# import third-party libraries
import numpy as np
from sympy import symbols, solve
from sympy.utilities.lambdify import lambdify

# import my libraries
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def get_top_side_bezier(param=[[0.6, 0.7], [0.8, 1.0], [1.0, 1.0]]):
    """
    Note
    ----
    An example of ```top_param``` is bellow.
        {'x0': 0.5, 'y0': 0.5,
         'x1': 0.7, 'y1': 0.7,
         'x2': 1.0, 'y2': 0.7}
    """
    a_val = param[0][0]
    b_val = param[1][0]
    c_val = param[2][0]
    p_val = param[0][1]
    q_val = param[1][1]
    r_val = param[2][1]

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


def youtube_linear(x):
    return 0.74 * x + 0.01175


def youtube_tonemapping(x, ks_luminance=400, ke_luminance=1000):
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
    ks_luminance : float
        knee start luminance. the unit is cd/m2.
    ke_luminance : float
        knee end luminance. the unit is cd/m2.
    """
    ks_x = tf.oetf_from_luminance(ks_luminance, tf.ST2084)
    ks_y = youtube_linear(ks_x)
    ke_x = tf.oetf_from_luminance(ke_luminance, tf.ST2084)
    ke_y = tf.oetf_from_luminance(100, tf.ST2084)
    mid_x = 0.6707140831310812
    mid_y = ke_y
    bezie = get_top_side_bezier([[ks_x, ks_y], [mid_x, mid_y], [ke_x, ke_y]])
    y = np.select(
        (x < ks_x, x <= ke_x, x > ke_x),
        (youtube_linear(x), bezie(x), ke_y))

    return y


if __name__ == '__main__':
    pass
