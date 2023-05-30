#ifndef SHOW_INTERNAL_VALUE_H
#define SHOW_INTERNAL_VALUE_H

#define FONT_COLOR_IDX_RED (0)
#define FONT_COLOR_IDX_GREEN (1)
#define FONT_COLOR_IDX_BLUE (2)
#define FONT_COLOR_IDX_CYAN (3)
#define FONT_COLOR_IDX_MAGENTA (4)
#define FONT_COLOR_IDX_YELLOW (5)

#define CROSS_HAIR_COLOR_IDX_RED (0)
#define CROSS_HAIR_COLOR_IDX_GREEN (1)
#define CROSS_HAIR_COLOR_IDX_BLUE (2)
#define CROSS_HAIR_COLOR_IDX_CYAN (3)
#define CROSS_HAIR_COLOR_IDX_MAGENTA (4)
#define CROSS_HAIR_COLOR_IDX_YELLOW (5)


__CONSTANT__ float3 rgbmyc_color[] = {
    {0.5, 0.0, 0.0},
    {0.0, 0.5, 0.0},
    {0.0, 0.0, 0.5},
    {0.0, 0.5, 0.5},
    {0.5, 0.0, 0.5},
    {0.5, 0.5, 0.0},
};

__CONSTANT__ float3 cross_hair_color = {0.5, 0.0, 0.5};
__CONSTANT__ float3 seven_seg_color = {0.5, 0.5, 0.5};
__CONSTANT__ float3 cross_hair_edge_color = {0.0, 0.0, 0.0};
__CONSTANT__ int digit_to_mask[] = {0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F};
#define TEXT_PERIOD_MASK (0x80)
#define TEXT_NEGATIVE_MASK (0x40)
#define TEXT_EFFECTIVE_DIGIT (6)


__DEVICE__ int draw_rectangle(int p_Width, int p_Height, int p_X, int p_Y, float3 *rgb, float2 st_pos, float2 ed_pos, float3 *line_color)
{
    if((st_pos.x <= p_X) && (p_X < ed_pos.x)){
        if((st_pos.y <= p_Y) && (p_Y < ed_pos.y)){
            rgb->x = line_color->x;
            rgb->y = line_color->y;
            rgb->z = line_color->z;
        }
    }

    return 0;
}


__DEVICE__ float2 calc_cross_hair_pos(int p_Width, int p_Height, int p_X, int p_Y, float h_center_pos_rate, float v_center_pos_rate)
{
    float2 pos;
    pos.x = h_center_pos_rate * p_Width + 0.5;
    pos.y = v_center_pos_rate * p_Height + 0.5;

    return pos;
}


__DEVICE__ float3 capture_rgb_value(int p_Width, int p_Height, int p_X, int p_Y, __TEXTURE__ p_TexR, __TEXTURE__ p_TexG, __TEXTURE__ p_TexB, float h_center_pos_rate, float v_center_pos_rate)
{
    float3 out;
    float2 center_pos = calc_cross_hair_pos(p_Width, p_Height, p_X, p_Y, h_center_pos_rate, v_center_pos_rate);
    out.x = _tex2D(p_TexR, int(center_pos.x), int(center_pos.y));
    out.y = _tex2D(p_TexG, int(center_pos.x), int(center_pos.y));
    out.z = _tex2D(p_TexB, int(center_pos.x), int(center_pos.y));

    return out;
}


__DEVICE__ int draw_cross_hair(int p_Width, int p_Height, int p_X, int p_Y, float3 *rgb, float h_center_pos_rate, float v_center_pos_rate, float3 *line_color)
{
    float cross_hair_rate = 0.04;
    float2 center_pos = calc_cross_hair_pos(p_Width, p_Height, p_X, p_Y, h_center_pos_rate, v_center_pos_rate);
    float3 *edge_color = &cross_hair_edge_color;
    float2 st_pos;
    float2 ed_pos;
    int line_width = int(_round(6 * (p_Height / 1080)));
    int edge_width = int(_round(12 * (p_Height / 1080)));
    int line_margin = int((edge_width - line_width) / 2.0);
    //int center_pos.x = int(h_center_pos_rate * p_Width + 0.5);
    int h_pos_st = int(center_pos.x - p_Height * cross_hair_rate);
    int h_pos_ed = int(center_pos.x + p_Height * cross_hair_rate);

    //int center_pos.y = int(v_center_pos_rate * p_Height + 0.5);
    int v_pos_st = int(center_pos.y - p_Height * cross_hair_rate);
    int v_pos_ed = int(center_pos.y + p_Height * cross_hair_rate);

    // top-edge
    st_pos.x = int(_round(center_pos.x - edge_width / 2.0));
    st_pos.y = v_pos_st;
    ed_pos.x = int(_round(center_pos.x + edge_width / 2.0));
    ed_pos.y = int(_round(center_pos.y - edge_width / 2.0));
    draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, edge_color);
    // top-line
    st_pos.x = st_pos.x + line_margin;
    st_pos.y = st_pos.y + line_margin;
    ed_pos.x = ed_pos.x - line_margin;
    ed_pos.y = ed_pos.y - line_margin;
    draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, line_color);

    // right-edge
    st_pos.x = int(_round(center_pos.x + edge_width / 2.0));
    st_pos.y = int(_round(center_pos.y - edge_width / 2.0));
    ed_pos.x = h_pos_ed;
    ed_pos.y = int(_round(center_pos.y + edge_width / 2.0));
    draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, edge_color);
    // right-line
    st_pos.x = st_pos.x + line_margin;
    st_pos.y = st_pos.y + line_margin;
    ed_pos.x = ed_pos.x - line_margin;
    ed_pos.y = ed_pos.y - line_margin;
    draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, line_color);

    // bottom-edge
    st_pos.x = int(_round(center_pos.x - edge_width / 2.0));
    st_pos.y = int(_round(center_pos.y + edge_width / 2.0));
    ed_pos.x = int(_round(center_pos.x + edge_width / 2.0));
    ed_pos.y = v_pos_ed;
    draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, edge_color);
    // bottom-line
    st_pos.x = st_pos.x + line_margin;
    st_pos.y = st_pos.y + line_margin;
    ed_pos.x = ed_pos.x - line_margin;
    ed_pos.y = ed_pos.y - line_margin;
    draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, line_color);

    // left-edge
    st_pos.x = h_pos_st;
    st_pos.y = int(_round(center_pos.y - edge_width / 2.0));
    ed_pos.x = int(_round(center_pos.x - edge_width / 2.0));
    ed_pos.y = int(_round(center_pos.y + edge_width / 2.0));
    draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, edge_color);
    // left-line
    st_pos.x = st_pos.x + line_margin;
    st_pos.y = st_pos.y + line_margin;
    ed_pos.x = ed_pos.x - line_margin;
    ed_pos.y = ed_pos.y - line_margin;
    draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, line_color);

//    // edge
//    if((h_pos_st - edge_width <= p_X) && (p_X < h_pos_ed + edge_width)){
//        if(((center_pos.y - line_width - edge_width) <= p_Y) && (p_Y < (center_pos.y + line_width + edge_width))){
//            rgb->x = edge_color->x;
//            rgb->y = edge_color->y;
//            rgb->z = edge_color->z;
//        }
//    }
//    if((v_pos_st - edge_width <= p_Y) && (p_Y < v_pos_ed + edge_width)){
//        if(((center_pos.x - line_width - edge_width) <= p_X) && (p_X < (center_pos.x + line_width + edge_width))){
//            rgb->x = edge_color->x;
//            rgb->y = edge_color->y;
//            rgb->z = edge_color->z;
//        }
//    }
//
//    // internal
//    if((h_pos_st <= p_X) && (p_X < h_pos_ed)){
//        if(((center_pos.y - line_width) <= p_Y) && (p_Y < (center_pos.y + line_width))){
//            rgb->x = line_color->x;
//            rgb->y = line_color->y;
//            rgb->z = line_color->z;
//        }
//    }
//
//    if((v_pos_st <= p_Y) && (p_Y < v_pos_ed)){
//        if(((center_pos.x - line_width) <= p_X) && (p_X < (center_pos.x + line_width))){
//            rgb->x = line_color->x;
//            rgb->y = line_color->y;
//            rgb->z = line_color->z;
//        }
//    }

    return 0;
}


__DEVICE__ int draw_single_digit(int p_Width, int p_Height, int p_X, int p_Y, float3 *rgb, float2 g_st_pos, int r_height, int r_width, int d_mask, float3 *line_color)
{
    float2 st_pos;
    float2 ed_pos;

    // A
    if((d_mask & 0x01) == 0x01){
        st_pos = make_float2(g_st_pos.x + r_height, g_st_pos.y);
        ed_pos = make_float2(st_pos.x + r_width, st_pos.y + r_height);
        draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, line_color);
    }

    // B
    if((d_mask & 0x02) == 0x02){
        st_pos = make_float2(g_st_pos.x + r_height + r_width, g_st_pos.y + r_height);
        ed_pos = make_float2(st_pos.x + r_height, st_pos.y + r_width);
        draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, line_color);
    }

    // C
    if((d_mask & 0x04) == 0x04){
        st_pos = make_float2(g_st_pos.x + r_height + r_width, g_st_pos.y + r_height * 2 + r_width);
        ed_pos = make_float2(st_pos.x + r_height, st_pos.y + r_width);
        draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, line_color);
    }

    // D
    if((d_mask & 0x08) == 0x08){
        st_pos = make_float2(g_st_pos.x + r_height, g_st_pos.y + (r_height + r_width) * 2);
        ed_pos = make_float2(st_pos.x + r_width, st_pos.y + r_height);
        draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, line_color);
    }

    // E
    if((d_mask & 0x10) == 0x10){
        st_pos = make_float2(g_st_pos.x, g_st_pos.y + r_height * 2 + r_width);
        ed_pos = make_float2(st_pos.x + r_height, st_pos.y + r_width);
        draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, line_color);
    }

    // F
    if((d_mask & 0x20) == 0x20){
        st_pos = make_float2(g_st_pos.x, g_st_pos.y + r_height);
        ed_pos = make_float2(st_pos.x + r_height, st_pos.y + r_width);
        draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, line_color);
    }

    // G
    if((d_mask & 0x40) == 0x40){
        st_pos = make_float2(g_st_pos.x + r_height, g_st_pos.y + r_height + r_width);
        ed_pos = make_float2(st_pos.x + r_width, st_pos.y + r_height);
        draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, line_color);
    }

    // H
    if((d_mask & 0x80) == 0x80){
        st_pos = make_float2(g_st_pos.x, g_st_pos.y + (r_height + r_width) * 2);
        ed_pos = make_float2(st_pos.x + r_height, st_pos.y + r_height);
        draw_rectangle(p_Width, p_Height, p_X, p_Y, rgb, st_pos, ed_pos, line_color);
    }

    return 0;
}


__DEVICE__ int calc_integer_digits(float drawing_value)
{
    int integer_digits = 1;
    int max_digits = TEXT_EFFECTIVE_DIGIT;
    int ii;

    for(ii=max_digits; ii > 0; ii--){
        if(drawing_value >= _powf(10, ii - 1)){
            integer_digits = ii;
            break;
        }
    }
    return integer_digits;
}


__DEVICE__ float2 calc_text_width(int r_width, int r_height)
{
    float2 out;
    out.x = r_width + r_height * 4;  // text width for digits
    out.y = r_height * 4;            // text width for period

    return out;
}


__DEVICE__ float2 draw_digits(int p_Width, int p_Height, int p_X, int p_Y, float3 *rgb, float drawing_value, float2 g_st_pos, int r_height, int r_width, float3 *font_color)
{
    int ii;
    float2 text_width_float = calc_text_width(r_width, r_height);
    int text_width = int(text_width_float.x);
    int text_width_period = int(text_width_float.y);
    float magnitude_value;
    int digit;
    int integer_digits;
    int decimal_digits;
    int drawing_value_int;
    int is_negative = 0;
    float2 st_pos = g_st_pos;

    if(drawing_value < 0){
        is_negative = 1;
    }
    drawing_value = _fabs(drawing_value);
    integer_digits = calc_integer_digits(drawing_value);
    decimal_digits = TEXT_EFFECTIVE_DIGIT - integer_digits;
    if(is_negative){
        decimal_digits -= 1;  // Using one digit to render the "-" character.
    }

    // convert float to int for round-up.
    drawing_value_int = int(_round(drawing_value * _powf(10, decimal_digits)));

    // for negative value
    if(is_negative){
        draw_single_digit(p_Width, p_Height, p_X, p_Y, rgb, st_pos, r_height, r_width, TEXT_NEGATIVE_MASK, font_color);
        st_pos.x += text_width;
    }

    // draw integer value
    st_pos.x -= text_width;  // To neutralize the effect of the initial offset calculation in the for loop.
    for(ii=0; ii<integer_digits; ii++){
        magnitude_value = _powf(10, (integer_digits + decimal_digits - ii - 1));
        digit = int(_fmod((_floorf(drawing_value_int / magnitude_value)), 10));
        st_pos.x += text_width;
        draw_single_digit(p_Width, p_Height, p_X, p_Y, rgb, st_pos, r_height, r_width, digit_to_mask[digit], font_color);
    }

    // draw "."
    st_pos.x += text_width;
    draw_single_digit(p_Width, p_Height, p_X, p_Y, rgb, st_pos, r_height, r_width, TEXT_PERIOD_MASK, font_color);
    st_pos.x += text_width_period;

    // draw decimal value
    for(ii=0; ii<decimal_digits; ii++){
        magnitude_value = _powf(10, decimal_digits - ii - 1);
        digit = int(_fmod((_floorf(drawing_value_int / magnitude_value)), 10));
        draw_single_digit(p_Width, p_Height, p_X, p_Y, rgb, st_pos, r_height, r_width, digit_to_mask[digit], font_color);
        st_pos.x += text_width;
    }

    st_pos.x += text_width;  // This is the margen between R, G, and B.

    return st_pos;
}


__DEVICE__ int draw_rgb_digits_core(int p_Width, int p_Height, int p_X, int p_Y, float3 *rgb, float3 *drawing_rgb_value, float2 g_st_pos, int r_height, int r_width, float3 *font_color)
{
    float2 st_pos = g_st_pos;
    float2 text_width_float = calc_text_width(r_width, r_height);
    int text_width = int(text_width_float.x);
    int text_width_period = int(text_width_float.y);
    int text_height = (r_height + r_width) * 2;
    int text_height_margin = r_height * 2;

    // draw background (dark)
    if((p_Y >= (g_st_pos.y - text_height_margin)) && (p_Y < (g_st_pos.y + text_height + text_height_margin))){
        if((p_X >= (g_st_pos.x - r_height * 2)) && (p_X < (g_st_pos.x + (text_width * TEXT_EFFECTIVE_DIGIT + text_width_period + text_width) * 3 - text_width))){
            rgb->x = rgb->x / 3.0;
            rgb->y = rgb->y / 3.0;
            rgb->z = rgb->z / 3.0;
        }
    }

    st_pos = draw_digits(p_Width, p_Height, p_X, p_Y, rgb, drawing_rgb_value->x, st_pos, r_height, r_width, font_color);
    st_pos = draw_digits(p_Width, p_Height, p_X, p_Y, rgb, drawing_rgb_value->y, st_pos, r_height, r_width, font_color);
    st_pos = draw_digits(p_Width, p_Height, p_X, p_Y, rgb, drawing_rgb_value->z, st_pos, r_height, r_width, font_color);
    return 0;
}


__DEVICE__ int draw_rgb_digits_with_cross_hair(int p_Width, int p_Height, int p_X, int p_Y, __TEXTURE__ p_TexR, __TEXTURE__ p_TexG, __TEXTURE__ p_TexB, float3 *out, int cross_hair_color_idx, int font_color_idx, float h_center_pos_rate, float v_center_pos_rate, float h_info_pos, float v_info_pos, float font_size_rate)
{

    float3 font_color = rgbmyc_color[font_color_idx];
    float3 cross_hair_color = rgbmyc_color[cross_hair_color_idx];

    float3 drawing_rgb_value = capture_rgb_value(p_Width, p_Height, p_X, p_Y, p_TexR, p_TexG, p_TexB, h_center_pos_rate, v_center_pos_rate);

    draw_cross_hair(p_Width, p_Height, p_X, p_Y, out, h_center_pos_rate, v_center_pos_rate, &cross_hair_color);

    float rectangle_width_rate = 0.05;
    float rectangle_height_rate = 0.01;
    float2 st_pos;
    float2 ed_pos;
    st_pos.x = p_Width * h_info_pos;
    st_pos.y = p_Height * v_info_pos;
    ed_pos.x = st_pos.x + p_Height * rectangle_width_rate;
    ed_pos.y = st_pos.y + p_Height * rectangle_height_rate;

    const float font_size_int_max = 100.0;  // It is the same value with max value of the `font_size_rate`
    int r_height = int(float(p_Height * font_size_rate) / (font_size_int_max * 60.0f) + 0.5f);
    int r_width = r_height * 5;
    int text_one_width = r_width + r_height * 4;
    int text_one_width_dot = r_height * 4;

    draw_rgb_digits_core(p_Width, p_Height, p_X, p_Y, out, &drawing_rgb_value, st_pos, r_height, r_width, &font_color);

    return 0;
}


#endif
