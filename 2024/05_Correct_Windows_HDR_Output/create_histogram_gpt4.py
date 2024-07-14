import os
import numpy as np
import pandas as pd
import plot_utility as pu


def plot_luminance_histogram():
    # CSVファイルを読み込む
    file_path = "./data/monitor_luminance_gamut_coverage.csv"
    data = pd.read_csv(file_path)

    # データの最初のいくつかの行とカラムの名前を確認
    data.head(), data.columns

    # 'HDR BRIGHTNESS Peak 10% Window' カラムから数値を抽出
    print(data)
    data['Brightness Values'] = data['HDR Brightness Peak 10% Window'].str.replace(' cd/m²', '').astype(float)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="HDR Peak Luminance (10% Window) Distribution",
        graph_title_size=None,
        xlabel="Luminance [nits]",
        ylabel="Frequency",
        axis_label_size=None,
        legend_size=17,
        xlim=[100, 1700],
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.grid(True, which='both', axis='y')
    ax1.grid(False, which='both', axis='x')
    ax1.hist(
        data['Brightness Values'],
        bins=range(150, int(data['Brightness Values'].max()) + 100, 100),
        color=pu.YELLOW, lw=2,
        edgecolor='black')
    pu.show_and_save(
        fig=fig, legend_loc=None,
        save_fname="./blog_img/monitor_luminance_distribution.png",
        show=False)


def plot_rec2020_coverage_histogram():
    # CSVファイルを読み込む
    file_path = "./data/monitor_luminance_gamut_coverage.csv"
    data = pd.read_csv(file_path)

    # 'HDR COLOR GAMUT Rec. 2020 Coverage xy' カラムから数値を再抽出して確認
    data['Gamut Coverage Values'] = data['HDR Color Gamut Rec.2020 Coverage'].str.replace('%', '').astype(float)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Rec.2020 Coverage Distribution",
        graph_title_size=None,
        xlabel="Coverage [%]",
        ylabel="Frequency",
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

    ax1.grid(True, which='both', axis='y')
    ax1.grid(False, which='both', axis='x')

    ax1.hist(
        data['Gamut Coverage Values'],
        bins=np.arange(50-(2.5/2), 85, 2.5),
        color=pu.SKY, lw=2,
        edgecolor='black')
    pu.show_and_save(
        fig=fig, legend_loc=None,
        save_fname="./blog_img/rec2020_coverage_distribution.png",
        show=True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_luminance_histogram()
    plot_rec2020_coverage_histogram()
