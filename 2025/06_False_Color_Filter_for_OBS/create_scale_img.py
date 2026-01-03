# import standard libraries
import os

# import third-party libraries
import numpy as np

# my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import font_control2 as fc2


def calc_pos_x_from_luminance(
        luminance, st_luminance, ed_luminance, s_height, margin):
    pos_x = round(
                (np.log10(luminance) - np.log10(st_luminance)) / \
                (np.log10(ed_luminance) - np.log10(st_luminance)) * s_height
            ) + margin
    
    return pos_x


def calc_luminance_line_pos(margin, s_width):
    st_pos_y = int(margin * 0.75)
    ed_pos_y = st_pos_y + s_width + int(margin * 0.5)

    return st_pos_y, ed_pos_y


def draw_luminance_line(
        luminance_list, st_luminance, ed_luminance, s_height, s_width, margin, img):
    color = tf.oetf_from_luminance(99.9, tf.ST2084)
    for luminance in luminance_list:
        pos_x = calc_pos_x_from_luminance(
            luminance=luminance, st_luminance=st_luminance, ed_luminance=ed_luminance,
            s_height=s_height, margin=margin
        )
        st_pos_y, ed_pos_y = calc_luminance_line_pos(margin=margin, s_width=s_width)
        img[st_pos_y:ed_pos_y, pos_x:pos_x+2] = color


def draw_luminance_text(
        luminance_list, st_luminance, ed_luminance, s_height, s_width, margin, img):
    g_height = img.shape[0]
    fg_cv = tf.oetf_from_luminance(95, tf.ST2084)
    fg_color = np.array([fg_cv, fg_cv, fg_cv])
    bg_color = np.array([0.0, 0.0, 0.0])
    font_size = 50


    for luminance in luminance_list:
        pos_y = g_height - calc_pos_x_from_luminance(
            luminance=luminance, st_luminance=st_luminance, ed_luminance=ed_luminance,
            s_height=s_height, margin=margin
        )
        _, line_ed_pos_x = calc_luminance_line_pos(margin=margin, s_width=s_width)
        text_pos_x = line_ed_pos_x + int(margin * 0.07)
        text_pos_y = pos_y

        # create instance
        text_draw_ctrl = fc2.TextDrawControl(
            text=f"{luminance}", font_color=fg_color,
            font_size=font_size, font_path=fc2.NOTO_SANS_CJKJP_BOLD,
            stroke_width=5, stroke_fill=bg_color)
        # calc position
        text_width, text_height = text_draw_ctrl.get_text_width_height()
        pos = (text_pos_x, text_pos_y - (text_height // 2) - (text_height // 20))
        # draw
        text_draw_ctrl.draw(img=img, pos=pos)


def create_scale_img():
    img_name = "./img/scale_img.png"
    g_height = 2160
    g_width = 300
    margin = int(g_height * 0.02)
    s_height = g_height - margin * 2
    s_width = 70

    st_luminance = 10
    ed_luminance = 10000

    luminance_list = [
        10, 100, 200, 400, 600, 1000, 2000, 4000, 10000
    ]

    img = np.ones((g_width, g_height, 3)) * np.array([0.0, 0.0001, 0.0])
    alpha = np.ones((g_height, g_width))

    x_st = np.log10(st_luminance)
    x_ed = np.log10(ed_luminance)
    x_log10 = np.linspace(x_st, x_ed, s_height)
    x_lumi = 10 ** x_log10
    x_cv = tf.oetf_from_luminance(x_lumi, tf.ST2084)

    bar_img = tpg.h_mono_line_to_img(line=x_cv, height=s_width)

    tpg.merge(img, bar_img, (margin, margin))

    # draw_luminance_line(
    #     luminance_list=luminance_list, st_luminance=st_luminance,
    #     ed_luminance=ed_luminance, s_height=s_height, s_width=s_width,
    #     margin=margin, img=img
    # )

    img = np.rot90(img)

    draw_luminance_text(
        luminance_list=luminance_list, st_luminance=st_luminance,
        ed_luminance=ed_luminance, s_height=s_height, s_width=s_width,
        margin=margin, img=img
    )

    mask = np.all(np.isclose(img, [0.0, 0.0001, 0.0]), axis=-1)
    alpha[mask] = 0.0
    img = np.dstack((img, alpha))

    print(img_name)
    tpg.img_wirte_float_as_16bit_int(img_name, img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_scale_img()
