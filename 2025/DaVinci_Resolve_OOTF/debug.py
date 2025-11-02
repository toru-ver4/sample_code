import os

import numpy as np
import test_pattern_generator2 as tpg
from colour import normalised_primary_matrix, matrix_RGB_to_RGB
import plot_utility as pu
import color_space as cs


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

    timeline_ramp = np.linspace(0, 1, len(x))

    e = non_linear_to_linear(timeline_ramp, gamma=output_gamma)
    forward_simulation = apply_forward_ootf(e)
    inverse_simulation = apply_inverse_ootf(e)
    
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
    

def eotf_BT1886(x : np.ndarray) -> np.ndarray:
    """
    Apply BT.1886 EOTF

    Parameters
    ----------
    x : np.ndarray
        Non linear code values.

    Returns
    -------
    np.ndarray
        Display referred linear light values.
    """
    return x ** 2.4


def eotf_inverse_BT1886(x : np.ndarray) -> np.ndarray:
    """
    Apply Inverse BT.1886 EOTF

    Parameters
    ----------
    x : np.ndarray
        Display referred linear light values.

    Returns
    -------
    np.ndarray
        Non linear code values.
    """
    return x ** (1/2.4)
    

def oetf_BT709(x : np.ndarray) -> np.ndarray:
    """
    Apply BT.709 OETF

    Parameters
    ----------
    x : np.ndarray
        Scene linear light values.

    Returns
    -------
    np.ndarray
        BT.709 encoded non linear code values.
    """
    y = np.where(x < 0.018, x * 4.5, 1.099 * (x ** 0.45) - 0.099)

    return y


def oetf_inverse_BT709(x : np.ndarray) -> np.ndarray:
    """
    Apply Inverse BT.709 OETF

    Parameters
    ----------
    x : np.ndarray
        BT.709 encoded non linear code values.

    Returns
    -------
    np.ndarray
        Scene linear light values.
    """
    y = np.where(
        oetf_BT709(0.018) > x,
        x / 4.5,
        ((x + 0.099) / 1.099) ** (1 / 0.45),
    )

    return y


def apply_forward_ootf(e : np.ndarray) -> np.ndarray:
    """
    Apply forward OOTF characteristics.

    Parameters
    ----------
    e : np.ndarray
        Scene referred linear value

    Returns
    -------
    np.ndarray
        Display referred linear values
    """
    e = np.asarray(e)

    return eotf_BT1886(oetf_BT709(e))


def apply_inverse_ootf(e : np.ndarray) -> np.ndarray:
    """
    Apply Inverse OOTF characteristics.

    Parameters
    ----------
    e : np.ndarray
        Display referred linear value

    Returns
    -------
    np.ndarray
        Scene referred linear values
    """
    e = np.asarray(e)

    return oetf_inverse_BT709((eotf_inverse_BT1886(e)))


def log_decoding_ARRILogC4(x : np.ndarray) -> np.ndarray:
    """
    Apply ARRILogC4 EOTF

    Parameters
    ----------
    x : np.ndarray
        LogC4 encoded non linear code value.

    Returns
    -------
    np.ndarray
        Scene linear light value.
    """
    a = (2**18 - 16) / 117.45
    b = (1023 - 95) / 1023
    c = 95 / 1023
    s = (7 * np.log(2) * 2 ** (7 - 14 * c / b)) / (a * b)
    t = (2 ** (14 * (-c / b) + 6) - 64) / a

    y = np.where(
        x >= 0,
        (2 ** (14 * ((x - c) / b) + 6) - 64) / a,
        x * s + t,
    )

    return y


def log_encoding_ARRILogC4(x: np.ndarray) -> np.ndarray:
    """
    Apply ARRILogC4 OETF

    Parameters
    ----------
    x : np.ndarray
        Scene linear light value.

    Returns
    -------
    np.ndarray
        LogC4 encoded non linear value.
    """
    a = (2**18 - 16) / 117.45
    b = (1023 - 95) / 1023
    c = 95 / 1023
    s = (7 * np.log(2) * 2 ** (7 - 14 * c / b)) / (a * b)
    t = (2 ** (14 * (-c / b) + 6) - 64) / a

    y = np.where(
        x >= t,
        (np.log2(a * x + 64) - 6) / 14 * b + c,
        (x - t) / s,
    )

    return y


def conv_logc4_to_gm26(x : np.ndarray) -> np.ndarray:
    """
    Apply LogC4 to Gamma 2.6 conversion without tone mapping.
    """
    linear = log_decoding_ARRILogC4(x)
    return np.clip(linear, 0.0, 1.0) ** (1/2.6)


def conv_gm26_to_logc4(x : np.ndarray) -> np.ndarray:
    """
    Apply Gamma 2.6 to LogC4 conversion without tone mapping.
    """
    linear = x ** 2.6
    return log_encoding_ARRILogC4(linear)


def conv_logc4_to_gm26_with_forward_ootf(x : np.ndarray) -> np.ndarray:
    """
    Apply LogC4 to Gamma 2.6 conversion with forward ootf without tone mapping.
    """
    linear = log_decoding_ARRILogC4(x)
    linear = np.clip(linear, 0.0, 1.0)

    linear_with_forward_ootf = apply_forward_ootf(linear)

    gamma26 = linear_with_forward_ootf ** (1/2.6)

    return gamma26


def conv_gm26_to_logc4_with_forward_ootf(x : np.ndarray) -> np.ndarray:
    """
    Apply Gamma 2.6 to LogC4 conversion with forward ootf without tone mapping.
    """
    linear = x ** 2.6

    linear_with_forward_ootf = apply_forward_ootf(linear)

    logc4 = log_encoding_ARRILogC4(linear_with_forward_ootf)

    return logc4


def conv_logc4_to_gm26_with_inverse_ootf(x : np.ndarray) -> np.ndarray:
    """
    Apply LogC4 to Gamma 2.6 conversion with inverse ootf without tone mapping.
    """
    linear = log_decoding_ARRILogC4(x)
    linear = np.clip(linear, 0.0, 1.0)

    linear_with_inverse_ootf = apply_inverse_ootf(linear)

    gamma26 = linear_with_inverse_ootf ** (1/2.6)

    return gamma26


def conv_gm26_to_logc4_with_inverse_ootf(x : np.ndarray) -> np.ndarray:
    """
    Apply 2.6 to Gamma LogC4 conversion with inverse ootf without tone mapping.
    """
    linear = x ** 2.6

    linear_with_inverse_ootf = apply_inverse_ootf(linear)

    logc4 = log_encoding_ARRILogC4(linear_with_inverse_ootf)

    return logc4


def debug2_wg4_logc4_to_p3d65_gm26():
    img_name_no_ootf = "./img/WG4_LogC4 to P3D65_GM26_no-ootf.png"
    img_name_forward_ootf = "./img/WG4_LogC4 to P3D65_GM26_forward-ootf.png"
    img_name_inverse_ootf = "./img/WG4_LogC4 to P3D65_GM26_inverse-ootf.png"

    img_no_ootf = tpg.img_read_as_float(img_name_no_ootf)
    img_forward_ootf = tpg.img_read_as_float(img_name_forward_ootf)
    img_inverse_ootf = tpg.img_read_as_float(img_name_inverse_ootf)


    # extract 1st channel for plot
    y_no_ootf = img_no_ootf[0, :, 0].reshape(-1)
    y_forward_ootf = img_forward_ootf[0, :, 0].reshape(-1)
    y_inverse_ootf = img_inverse_ootf[0, :, 0].reshape(-1)

    x = np.linspace(0, 1, len(y_no_ootf))
    my_no_ootf = conv_logc4_to_gm26(x)
    my_forward_ootf = conv_logc4_to_gm26_with_forward_ootf(x)
    my_inverse_ootf = conv_logc4_to_gm26_with_inverse_ootf(x)

    fig, ax1 = pu.plot_1_graph(
        fontsize=14,
        figsize=(12, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="ARRI LogC4 to P3D65 Gamma 2.6",
        graph_title_size=None,
        xlabel="Input Code Value (LogC4)",
        ylabel="Output Code Value (Gamma 2.6)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        # xtick=[x * 64 for x in range(1024//64)] + [1023],
        # ytick=[x * 128 for x in range(1024//128)] + [1023],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y_no_ootf, color=pu.RED, label="Resolve: No-OOTF")
    ax1.plot(x, y_forward_ootf, color=pu.GREEN, label="Resolve: Forward-OOTF")
    ax1.plot(x, y_inverse_ootf, color=pu.BLUE, label="Resolve: Inverse-OOTF")

    ax1.plot(x, my_no_ootf, '--', color=pu.PINK, label="Python: No-OOTF")
    ax1.plot(x, my_forward_ootf, '--', color=pu.BROWN, label='Python: Forward-OOTF')
    ax1.plot(x, my_inverse_ootf, '--', color=pu.SKY, label='Python: Inverse-OOTF')

    fname = "./img/all_graph_logc4_to_gm26.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', fontsize=16, save_fname=fname, show=True)


def debug2_p3d65_gm26_to_wg4_logc4():
    img_name_no_ootf = "./img/P3D65_GM26_to_WG4_LogC4_no-ootf.png"
    img_name_forward_ootf = "./img/P3D65_GM26_to_WG4_LogC4_forward-ootf.png"
    img_name_inverse_ootf = "./img/P3D65_GM26_to_WG4_LogC4_inverse-ootf.png"

    img_no_ootf = tpg.img_read_as_float(img_name_no_ootf)
    img_forward_ootf = tpg.img_read_as_float(img_name_forward_ootf)
    img_inverse_ootf = tpg.img_read_as_float(img_name_inverse_ootf)


    # extract 1st channel for plot
    y_no_ootf = img_no_ootf[0, :, 0].reshape(-1)
    y_forward_ootf = img_forward_ootf[0, :, 0].reshape(-1)
    y_inverse_ootf = img_inverse_ootf[0, :, 0].reshape(-1)

    x = np.linspace(0, 1, len(y_no_ootf))
    my_no_ootf = conv_gm26_to_logc4(x)
    my_forward_ootf = conv_gm26_to_logc4_with_forward_ootf(x)
    my_inverse_ootf = conv_gm26_to_logc4_with_inverse_ootf(x)

    fig, ax1 = pu.plot_1_graph(
        fontsize=14,
        figsize=(12, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="P3D65 Gamma 2.6 to ARRI LogC4",
        graph_title_size=None,
        xlabel="Input Code Value (Gamma 2.6)",
        ylabel="Output Code Value (LogC4)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        # xtick=[x * 64 for x in range(1024//64)] + [1023],
        # ytick=[x * 128 for x in range(1024//128)] + [1023],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y_no_ootf, color=pu.RED, label="Resolve: No-OOTF")
    ax1.plot(x, y_forward_ootf, color=pu.GREEN, label="Resolve: Forward-OOTF")
    ax1.plot(x, y_inverse_ootf, color=pu.BLUE, label="Resolve: Inverse-OOTF")

    ax1.plot(x, my_no_ootf, '--', color=pu.PINK, label="Python: No-OOTF")
    ax1.plot(x, my_forward_ootf, '--', color=pu.BROWN, label='Python: Forward-OOTF')
    ax1.plot(x, my_inverse_ootf, '--', color=pu.SKY, label='Python: Inverse-OOTF')

    fname = "./img/all_graph_gm26_to_logc4.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', fontsize=16, save_fname=fname, show=True)
    

def calc_arri_wg4_to_p3d65_matrix():
    arri_wg4_primaries = cs.get_primaries(cs.ALEXA_WIDE_GAMUT_4)
    p3d65_primaries = cs.get_primaries(cs.P3_D65)
    
    arri_wg4_to_xyz = normalised_primary_matrix(arri_wg4_primaries, cs.D65)
    p3d65_to_xyz = normalised_primary_matrix(p3d65_primaries, cs.D65)
    xyz_to_p3d65 = np.linalg.inv(p3d65_to_xyz)

    arri_wg4_to_p3d65 = xyz_to_p3d65.dot(arri_wg4_to_xyz)

    return arri_wg4_to_p3d65


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug1_plot_three_graph(output_gamma=2.4)
    # debug1_plot_three_graph(output_gamma=2.2)
    # debug1_plot_three_graph(output_gamma="Rec.709")
    # debug2_wg4_logc4_to_p3d65_gm26()
    # debug2_p3d65_gm26_to_wg4_logc4()
    print(calc_arri_wg4_to_p3d65_matrix())
