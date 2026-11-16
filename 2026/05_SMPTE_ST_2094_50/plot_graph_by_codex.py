"""Draw the endpoint conditions for cubic Hermite interpolation.

Run from the repository root:
    .venv/Scripts/python.exe 2026/05_SMPTE_ST_2094_50/plot_graph_by_codex.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from study_pchip import my_plot_quadratic_hermite_spline


def plot_hermite_endpoint_conditions():
    """Save the endpoint conditions of f(x) as a PNG.

    Returns
    -------
    None

    Notes
    -----
    Coordinates and slopes are illustrative. Only the endpoint conditions
    are shown, as the first step in explaining cubic Hermite interpolation.
    The output directory is resolved relative to this script.

    Examples
    --------
    >>> plot_hermite_endpoint_conditions()
    """
    _plot_hermite_endpoint_conditions(normalized=False)


def plot_normalized_hermite_endpoint_conditions():
    """Save the endpoint conditions of the normalized function g(t).

    Returns
    -------
    None

    Notes
    -----
    With h = x_{i+1} - x_i, g(t) = f(x_i + h*t). Endpoint values
    are unchanged, while derivatives with respect to t equal h times
    the corresponding derivatives of f(x). The style matches the f(x) plot.

    Examples
    --------
    >>> plot_normalized_hermite_endpoint_conditions()
    """
    _plot_hermite_endpoint_conditions(normalized=True)


def plot_hermite_endpoint_conditions_with_interpolation():
    """Save f(x) with its Hermite interpolant between the endpoints.

    Returns
    -------
    None

    Examples
    --------
    >>> plot_hermite_endpoint_conditions_with_interpolation()
    """
    _plot_hermite_endpoint_conditions(normalized=False, interpolate=True)


def plot_normalized_hermite_endpoint_conditions_with_interpolation():
    """Save g(t) with its Hermite interpolant from t=0 to t=1.

    Returns
    -------
    None

    Examples
    --------
    >>> plot_normalized_hermite_endpoint_conditions_with_interpolation()
    """
    _plot_hermite_endpoint_conditions(normalized=True, interpolate=True)


def _plot_hermite_endpoint_conditions(normalized, interpolate=False):
    """Draw endpoint conditions in the original or normalized coordinates.

    Parameters
    ----------
    normalized : bool
        Whether to use t = (x - x_i) / (x_{i+1} - x_i).
    interpolate : bool, optional
        Add the Hermite interpolant between the endpoints.

    Returns
    -------
    None

    Examples
    --------
    >>> _plot_hermite_endpoint_conditions(normalized=True)
    """
    filename = (
        "hermite_endpoint_conditions_normalized.png"
        if normalized else "hermite_endpoint_conditions.png"
    )
    if interpolate:
        filename = filename.replace('.png', '_interpolated.png')
    output_path = (
        Path(__file__).resolve().parent
        / "img_by_codex" / filename
    )
    background = "#20201f"
    foreground = "#eeeeee"
    tangent_color = "#e5ed00"
    x_i, x_ip1 = 1.25, 3.75
    y_i, y_ip1 = 0.8, 2.2
    m_i, m_ip1 = 1.5, -1.0
    h = x_ip1 - x_i
    if normalized:
        x_i, x_ip1 = 0.0, 1.0
        m_i, m_ip1 = h * m_i, h * m_ip1
    horizontal_scale = 1 / h if normalized else 1.0

    def horizontal(value):
        """Scale a horizontal distance to the selected coordinates.

        Parameters
        ----------
        value : float
            Distance in the original coordinates.

        Returns
        -------
        float
            Distance in the selected coordinates.

        Examples
        --------
        >>> horizontal(0.0)
        0.0
        """
        return value * horizontal_scale

    with plt.rc_context({"mathtext.fontset": "cm", "font.size": 24}):
        fig, ax = plt.subplots(figsize=(10.8, 6), facecolor=background)
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.07, top=0.95)
        ax.set_facecolor(background)
        ax.set_xlim((-0.34, 2.24) if normalized else (-0.85, 5.6))
        ax.set_ylim(-0.65, 3.25)
        ax.axis("off")

        if interpolate:
            # Normalized endpoints already have slopes scaled by h.
            my_plot_quadratic_hermite_spline(
                x_i0=x_i, x_i1=x_ip1, y_i0=y_i, y_i1=y_ip1,
                m_i0=m_i, m_i1=m_ip1, ax=ax, normalized=normalized,
                color="#55cfff", linewidth=2.5, zorder=2,
            )

        # Arrow axes and symbolic ticks reproduce the reference sketch.
        arrow_style = {"arrowstyle": "->", "color": foreground,
                       "linewidth": 1.5, "mutation_scale": 19}
        ax.annotate("", xy=(1.4 if normalized else 5.0, 0),
                    xytext=(horizontal(-0.4), 0),
                    arrowprops=arrow_style)
        ax.text(1.45 if normalized else 5.125, 0,
                r"$t$" if normalized else r"$x$", color=foreground,
                ha="left", va="center")
        ax.annotate("", xy=(0, 3.0),
                    xytext=(0, -0.15 if normalized else -0.45),
                    arrowprops=arrow_style)
        ax.text(0, 3.08, r"$y$", color=foreground,
                ha="center", va="bottom")

        ticks = (
            ((x_i, r"$0$"), (x_ip1, r"$1$")) if normalized else
            ((x_i, r"$x_i$"), (x_ip1, r"$x_{i+1}$"))
        )
        for x, label in ticks:
            ax.plot([x, x], [-0.09, 0.09], color=foreground, linewidth=1.2)
            ax.text(x, -0.25, label, color=foreground,
                    ha="center", va="top")

        for x, y, label in (
            (x_i, y_i, r"$y_i$"), (x_ip1, y_ip1, r"$y_{i+1}$")
        ):
            ax.plot([0, x], [y, y], color=foreground,
                    linewidth=1.2, linestyle=(0, (5, 8)))
            ax.text(horizontal(-0.22), y, label, color=foreground,
                    ha="right", va="center")

        for x, y in ((x_i, y_i), (x_ip1, y_ip1)):
            ax.plot([x, x], [0, y], color=foreground,
                    linewidth=1.2, linestyle=(0, (5, 8)))

        # Each short line passes through its endpoint with the given slope.
        half_width = horizontal(0.22)
        for x, y, slope in ((x_i, y_i, m_i), (x_ip1, y_ip1, m_ip1)):
            ax.plot([x - half_width, x + half_width],
                    [y - slope * half_width, y + slope * half_width],
                    color=tangent_color, linewidth=1.8,
                    solid_capstyle="round", zorder=3)
            ax.plot(x, y, marker="o", markersize=4,
                    markerfacecolor=background, markeredgecolor=foreground,
                    markeredgewidth=1, zorder=4)

        start_label = (
            r"$g^\prime(0)$" if normalized else
            r"$f^\prime(x_i)=m_i$"
        )
        end_label = (
            r"$g^\prime(1)$" if normalized else
            r"$f^\prime(x_{i+1})=m_{i+1}$"
        )
        ax.text(x_i + horizontal(0.28), y_i, start_label,
                color=tangent_color, va="center")
        ax.text(x_ip1 + horizontal(0.25), y_ip1 + 0.28, end_label,
                color=tangent_color, va="center")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, facecolor=background)
        plt.close(fig)

    print(f"Saved: {output_path}")


def plot_pchip_base_condition():
    """Save a diagram of the values, derivatives, and secant slopes for PCHIP.

    Returns
    -------
    None

    Notes
    -----
    Following Fritsch and Butland (1984), f_i denotes a nodal value,
    d_i = p'(x_i) a nodal derivative, and Delta_i the secant slope
    (f_{i+1} - f_i) / (x_{i+1} - x_i). The illustrative monotone curve
    is quadratic, so its values and derivatives are mutually consistent.
    Save the figure to ``img_by_codex/pchip_base_condition.png``.

    Examples
    --------
    >>> plot_pchip_base_condition()
    """
    output_path = (
        Path(__file__).resolve().parent
        / "img_by_codex" / "pchip_base_condition.png"
    )
    background = "#20201f"
    foreground = "#eeeeee"
    tangent_color = "#ff354e"
    secant_color = "#e5ed00"
    x = np.array([1.0, 2.9, 4.3])
    y = 3.15 - 0.2 * (4.3 - x)**2
    derivatives = 0.4 * (4.3 - x)
    curve_x = np.linspace(0.65, x[-1], 500)
    curve_y = 3.15 - 0.2 * (4.3 - curve_x)**2

    with plt.rc_context({"mathtext.fontset": "cm", "font.size": 28}):
        fig, ax = plt.subplots(figsize=(11, 8.5), facecolor=background)
        fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.98)
        ax.set_facecolor(background)
        ax.set_xlim(-0.85, 5.35)
        ax.set_ylim(-0.55, 4.4)
        ax.axis("off")

        arrow_style = {"arrowstyle": "->", "color": foreground,
                       "linewidth": 1.6, "mutation_scale": 22}
        ax.annotate("", xy=(5.05, 0), xytext=(-0.22, 0),
                    arrowprops=arrow_style)
        ax.annotate("", xy=(0, 3.85), xytext=(0, -0.25),
                    arrowprops=arrow_style)
        ax.text(5.13, 0, r"$x$", color=foreground, va="center")
        annotation_x = 1.75
        annotation_y = 3.15 - 0.2 * (4.3 - annotation_x)**2
        ax.annotate(
            r"$p(x)$", xy=(annotation_x, annotation_y),
            xytext=(0.7, 2.4), color=foreground, ha="center", va="center",
            arrowprops={"arrowstyle": "->", "color": foreground,
                        "linewidth": 1.6, "mutation_scale": 22,
                        "connectionstyle": "arc3,rad=-0.15", "shrinkB": 7},
        )

        indices = ("i-1", "i", "i+1")
        for xi, yi, index in zip(x, y, indices):
            ax.plot([0, xi], [yi, yi], color=foreground,
                    linewidth=1.1, linestyle=(0, (6, 7)), alpha=0.65)
            ax.plot([xi, xi], [0, yi], color=foreground,
                    linewidth=1.1, linestyle=(0, (6, 7)), alpha=0.65)
            ax.plot([xi, xi], [-0.055, 0.055], color=foreground,
                    linewidth=1.3)
            ax.text(xi, -0.17, rf"$x_{{{index}}}$", color=foreground,
                    ha="center", va="top")
            ax.text(-0.16, yi, rf"$f_{{{index}}}$", color=foreground,
                    ha="right", va="center")

        ax.plot(curve_x, curve_y, color=foreground, linewidth=2.5, zorder=2)
        ax.plot(x, y, color=secant_color, linewidth=1.8, zorder=3)
        for xi, yi, slope, index in zip(x, y, derivatives, indices):
            half_width = 0.23
            ax.plot([xi - half_width, xi + half_width],
                    [yi - slope * half_width, yi + slope * half_width],
                    color=tangent_color, linewidth=2.4,
                    solid_capstyle="round", zorder=4)
            ax.plot(xi, yi, "o", markersize=5, color=foreground, zorder=5)

        for xi, yi, label in (
            (x[0] - 0.32, y[0] + 0.27, r"$d_{i-1}$"),
            (x[1] - 0.11, y[1] + 0.26, r"$d_i$"),
            (x[2] + 0.08, y[2] + 0.26, r"$d_{i+1}$"),
        ):
            ax.text(xi, yi, label, color=tangent_color,
                    ha="center", va="center")
        ax.text(1.95, 1.54, r"$\Delta_{i-1}$", color=secant_color,
                ha="center", va="center")
        ax.text(3.63, 2.69, r"$\Delta_i$", color=secant_color,
                ha="center", va="center")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, facecolor=background)
        plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    # plot_hermite_endpoint_conditions()
    # plot_normalized_hermite_endpoint_conditions()
    # plot_hermite_endpoint_conditions_with_interpolation()
    # plot_normalized_hermite_endpoint_conditions_with_interpolation()

    plot_pchip_base_condition()
