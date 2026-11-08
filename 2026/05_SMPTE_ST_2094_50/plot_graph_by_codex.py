"""Draw the endpoint conditions for cubic Hermite interpolation.

Run from the repository root:
    .venv/Scripts/python.exe 2026/05_SMPTE_ST_2094_50/plot_graph_by_codex.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    plot_hermite_endpoint_conditions()
    plot_normalized_hermite_endpoint_conditions()
    plot_hermite_endpoint_conditions_with_interpolation()
    plot_normalized_hermite_endpoint_conditions_with_interpolation()
