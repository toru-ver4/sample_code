import os

import numpy as np
from colour.models import oetf_BT709, oetf_inverse_BT709

import test_pattern_generator2 as tpg
import transfer_functions as tf
import plot_utility as pu


def debug1_plot_three_graph(output_gamma: float | str = 2.4):
    if isinstance(output_gamma, str):
        output_gamma_str = output_gamma
    else:
        output_gamma_str = f"{output_gamma:.1f}"
    no_ootf_fname = f"./img/Rec.709_Gamma{output_gamma_str}_No-OOTF.png"
    forward_ootf_fname = f"./img/Rec.709_Gamma{output_gamma_str}_Forward-OOTF.png"
    inverse_ootf_fname = f"./img/Rec.709_Gamma{output_gamma_str}_Inverse-OOTF.png"

    fname_list = [no_ootf_fname, forward_ootf_fname, inverse_ootf_fname]
    label_list = ["No OOTF", "Foward OOTF", "Inverse OOTF"]
    color_list = [pu.RED, pu.GREEN, pu.BLUE]


    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"OOTF Analysis (Gamma = {output_gamma_str})",
        graph_title_size=None,
        xlabel=f"Input Code Value (Gamma = {output_gamma_str})",
        ylabel=f"Output Code Value (Gamma = {output_gamma_str})",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    for idx in range(len(fname_list)):
        fname = fname_list[idx]
        label = label_list[idx]
        color = color_list[idx]
        img = tpg.img_read_as_float(fname)
        line_data = img[0, :, 1].reshape(-1)
        x = np.linspace(0, 1, len(line_data))
        ax1.plot(x, line_data, color=color, label=label)
    x = np.linspace(0, 1, 1024)
    bt1886_gamma = 2.4
    e = gamma_func(x, gamma=output_gamma)
    forward_simulation = gamma_inv_func((oetf_BT709(e) ** bt1886_gamma), gamma=output_gamma)
    inverse_simulation = gamma_inv_func(oetf_inverse_BT709((e ** (1/bt1886_gamma))), gamma=output_gamma)
    ax1.plot(x, inverse_simulation, '--', color=pu.PINK, lw=2, label="Inverse Simulation")
    ax1.plot(x, forward_simulation, '--', color=pu.BROWN, lw=2, label="Forward Simulation")
    graph_fname = f"./img/all_graph_gamma_{output_gamma_str}.png"
    print(graph_fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=graph_fname, show=False
    )


def gamma_inv_func(x, gamma: float | str = 2.4):
    if isinstance(gamma, str):
        return oetf_BT709(x)
    else:
        return x ** (1/gamma)
    

def gamma_func(x, gamma: float | str = 2.4):
    if isinstance(gamma, str):
        return oetf_inverse_BT709(x)
    else:
        return x ** (gamma)


def maybe_forward_ootf_simulation():
    linear_light = np.linspace(0, 1, 2 ** 16)
    y = oetf_BT709(linear_light)

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
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(linear_light, linear_light, label="Reference")
    ax1.plot(linear_light, y, label="Rec.709 OETF")
    ax1.plot(linear_light, y ** 2.4, label="Rec.709 OETF + Rec.1886 EOTF")
    graph_fname = None
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=graph_fname, show=True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug1_plot_three_graph(output_gamma=2.4)
    debug1_plot_three_graph(output_gamma=2.2)
    debug1_plot_three_graph(output_gamma="Rec.709")
    # maybe_forward_ootf_simulation()
