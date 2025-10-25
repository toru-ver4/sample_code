import os

import numpy as np
from colour.models import oetf_BT709, oetf_inverse_BT709, eotf_BT1886, eotf_inverse_BT1886
import test_pattern_generator2 as tpg
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
    label_list = ["Resolve: No OOTF", "Resolve: Foward OOTF", "Resolve: Inverse OOTF"]
    color_list = [pu.GREEN, pu.RED, pu.BLUE]


    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"OOTF Characteristics (Gamma = {output_gamma_str})",
        graph_title_size=None,
        xlabel=f"Timeline Code Value (Gamma = {output_gamma_str})",
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

    timeline_ramp = np.linspace(0, 1, 1024)

    e = non_linear_to_linear(timeline_ramp, gamma=output_gamma)
    forward_simulation = eotf_BT1886(oetf_BT709(e))
    inverse_simulation = oetf_inverse_BT709((eotf_inverse_BT1886(e)))
    
    forward_simulation_non_linear = linear_to_non_linear(forward_simulation, gamma=output_gamma)
    inverse_simulation_non_linear = linear_to_non_linear(inverse_simulation, gamma=output_gamma)
    ax1.plot(x, forward_simulation_non_linear, ':', color=pu.MAJENTA, lw=2, label="Python: Forward Simulation")
    ax1.plot(x, inverse_simulation_non_linear, ':', color=pu.SKY, lw=2, label="Python: Inverse Simulation")
    graph_fname = f"./img/all_graph_gamma_{output_gamma_str}.png"
    print(graph_fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=graph_fname, show=False
    )


def linear_to_non_linear(x, gamma: float | str = 2.4):
    if isinstance(gamma, str):
        return oetf_BT709(x)
    else:
        return x ** (1/gamma)
    

def non_linear_to_linear(x, gamma: float | str = 2.4):
    if isinstance(gamma, str):
        return oetf_inverse_BT709(x)
    else:
        return x ** (gamma)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug1_plot_three_graph(output_gamma=2.4)
    debug1_plot_three_graph(output_gamma=2.2)
    debug1_plot_three_graph(output_gamma="Rec.709")
