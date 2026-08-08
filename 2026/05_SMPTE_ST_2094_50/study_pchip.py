#!/usr/bin/env python
# import standard libraries
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# import third-party libraries
from sympy import symbols

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2026 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def study_cubic_hermite_polynominal():
    x, x_i, x_ip1, y_i, y_ip1, m_i, m_1p1, c_3, c_2, c_1, c_0 = symbols('x, x_i, x_{x+1}, y_i, y_{i+1}, m_i, m_{i+1}, c_3, c_2, c_1, c_0')
    h = x_ip1 - x_i
    t = (x - x_i) / h
    f = c_3 * (t ** 3) + c_2 * (t ** 2) + c_1 * t + c_0
    coef_0 = y_i
    coef_1 = h * m_i
    coef_2 = -3 * y_i + 3 * y_ip1 - 2 * h * m_i - h * m_1p1
    coef_3 = 2 * y_i - 2 * y_ip1 + h * m_i + h * m_1p1
    print(f.subs({c_0: coef_0, c_1: coef_1, c_2: coef_2, c_3: coef_3}))


def calc_harmonic_mean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    numerator = 2 * x * y
    denominator = (x + y)

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def plot_harmonic_mean():
    """Plot the harmonic and arithmetic means in the same 3D axes.

    Returns
    -------
    None

    Examples
    --------
    >>> plot_harmonic_mean()
    """
    samples = 51
    x = np.linspace(0, 1, samples)
    y = np.linspace(0, 1, samples)
    xx, yy = np.meshgrid(x, y)
    h_mean = calc_harmonic_mean(xx, yy)
    normal_mean = (xx + yy) / 2

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surface = ax.plot_surface(
        xx, yy, h_mean, cmap="Blues", alpha=0.85, edgecolor="none"
    )
    ax.plot_wireframe(
        xx, yy, normal_mean, rstride=5, cstride=5,
        color="tab:orange", linewidth=1.0
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Mean")
    ax.set_title("Harmonic mean vs. arithmetic mean")
    ax.legend(handles=[
        Patch(facecolor="tab:blue", alpha=0.85, label="Harmonic mean"),
        Line2D([0], [0], color="tab:orange", label="Arithmetic mean"),
    ])
    fig.colorbar(surface, ax=ax, shrink=0.7, label="Harmonic mean")
    fig.tight_layout()
    plt.show()


def plot_marmonic_mean_diff():
    """Plot the difference between the arithmetic and harmonic means.

    Returns
    -------
    None

    Examples
    --------
    >>> plot_marmonic_mean_diff()
    """
    samples = 51
    x = np.linspace(0, 1, samples)
    y = np.linspace(0, 1, samples)
    xx, yy = np.meshgrid(x, y)
    h_mean = calc_harmonic_mean(xx, yy)
    normal_mean = (xx + yy) / 2
    mean_diff = normal_mean - h_mean

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surface = ax.plot_surface(
        xx, yy, mean_diff, cmap="magma", edgecolor="none"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Arithmetic mean - harmonic mean")
    ax.set_title("Difference between arithmetic and harmonic means")
    fig.colorbar(
        surface, ax=ax, shrink=0.7,
        label="Arithmetic mean - harmonic mean"
    )
    fig.tight_layout()
    plt.show()


def plot_cubi_hermite_spline_explanation():
    """Plot the values and slopes used for cubic Hermite interpolation.

    Returns
    -------
    None

    Notes
    -----
    The displayed cubic has an inflection point inside the interpolation
    interval. Its endpoint slopes have opposite signs.

    Examples
    --------
    >>> plot_cubi_hermite_spline_explanation()
    """
    x_i = 1.0
    x_ip1 = 4.0

    def f(x):
        """Evaluate the cubic polynomial used in the illustration.

        Parameters
        ----------
        x : float or numpy.ndarray
            Coordinate at which to evaluate the polynomial.

        Returns
        -------
        float or numpy.ndarray
            Polynomial value at ``x``.

        Examples
        --------
        >>> f(1.0)
        1.3
        """
        t = (x - x_i) / (x_ip1 - x_i)
        return 1.3 - 2.0 * t + 8.0 * t**2 - 4.0 * t**3

    def derivative(x):
        """Evaluate the derivative of the illustrated polynomial.

        Parameters
        ----------
        x : float or numpy.ndarray
            Coordinate at which to evaluate the derivative.

        Returns
        -------
        float or numpy.ndarray
            Derivative value at ``x``.

        Examples
        --------
        >>> derivative(1.0)
        -0.6666666666666666
        """
        t = (x - x_i) / (x_ip1 - x_i)
        return (-2.0 + 16.0 * t - 12.0 * t**2) / (x_ip1 - x_i)

    y_i = f(x_i)
    y_ip1 = f(x_ip1)
    m_i = derivative(x_i)
    m_ip1 = derivative(x_ip1)

    x = np.linspace(0.45, 4.55, 401)
    x_interval = np.linspace(x_i, x_ip1, 301)
    tangent_width = 0.72
    tangent_i_x = np.array([x_i - tangent_width, x_i + tangent_width])
    tangent_ip1_x = np.array([
        x_ip1 - tangent_width, x_ip1 + tangent_width
    ])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, f(x), color="0.72", linewidth=1.6)
    ax.plot(
        x_interval, f(x_interval), color="tab:blue", linewidth=3.0
    )
    ax.plot(
        tangent_i_x, y_i + m_i * (tangent_i_x - x_i),
        color="tab:orange", linewidth=2.0
    )
    ax.plot(
        tangent_ip1_x, y_ip1 + m_ip1 * (tangent_ip1_x - x_ip1),
        color="tab:green", linewidth=2.0
    )
    ax.scatter(
        [x_i, x_ip1], [y_i, y_ip1], s=55, color="black", zorder=5
    )

    ax.vlines(
        [x_i, x_ip1], 0, [y_i, y_ip1],
        colors="0.55", linestyles="dashed", linewidth=1.0
    )
    ax.hlines(
        [y_i, y_ip1], 0, [x_i, x_ip1],
        colors="0.55", linestyles="dashed", linewidth=1.0
    )

    annotation_style = dict(
        arrowprops=dict(arrowstyle="->", color="0.25", linewidth=1.0),
        fontsize=14,
        fontweight="bold",
        fontfamily="serif",
        math_fontfamily="cm",
    )
    ax.annotate(
        r"$\boldsymbol{m_i=f^\prime(x_i)<0}$",
        xy=(x_i, y_i), xytext=(1.55, 0.72),
        color="tab:orange", **annotation_style
    )
    ax.annotate(
        r"$\boldsymbol{m_{i+1}=f^\prime(x_{i+1})>0}$",
        xy=(x_ip1, y_ip1), xytext=(2.05, 3.85),
        color="tab:green", **annotation_style
    )

    ax.set_xlim(0, 4.8)
    ax.set_ylim(0, 4.25)
    ax.set_xticks([x_i, x_ip1], labels=[r"$x_i$", r"$x_{i+1}$"])
    ax.set_yticks([y_i, y_ip1], labels=[r"$y_i$", r"$y_{i+1}$"])
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontfamily("serif")
        tick_label.set_math_fontfamily("cm")
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlabel(
        r"$x$", fontsize=18, fontfamily="serif", math_fontfamily="cm"
    )
    ax.set_ylabel(
        r"$y=f(x)$", fontsize=18, fontfamily="serif", math_fontfamily="cm"
    )
    ax.set_title("Quantities used in cubic Hermite interpolation")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # study_cubic_hermite_polynominal()
    # plot_harmonic_mean()
    # plot_marmonic_mean_diff()
    plot_cubi_hermite_spline_explanation()
