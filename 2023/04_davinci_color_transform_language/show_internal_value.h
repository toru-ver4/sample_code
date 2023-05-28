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
__CONSTANT__ int digit_to_mask[] = {0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F};
#define TEXT_PERIOD_MASK (0x80)
#define TEXT_EFFECTIVE_DIGIT (6)


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
    float cross_hair_rate = 0.025;
    float2 center_pos = calc_cross_hair_pos(p_Width, p_Height, p_X, p_Y, h_center_pos_rate, v_center_pos_rate);
    int line_width = 3;
    //int center_pos.x = int(h_center_pos_rate * p_Width + 0.5);
    int h_pos_st = int(center_pos.x - p_Height * cross_hair_rate);
    int h_pos_ed = int(center_pos.x + p_Height * cross_hair_rate);
    //int center_pos.y = int(v_center_pos_rate * p_Height + 0.5);
    int v_pos_st = int(center_pos.y - p_Height * cross_hair_rate);
    int v_pos_ed = int(center_pos.y + p_Height * cross_hair_rate);

    // h-line
    if((h_pos_st <= p_X) && (p_X < h_pos_ed)){
        if(((center_pos.y - line_width) <= p_Y) && (p_Y < (center_pos.y + line_width))){
            rgb->x = line_color->x;
            rgb->y = line_color->y;
            rgb->z = line_color->z;
        }
    }

    if((v_pos_st <= p_Y) && (p_Y < v_pos_ed)){
        if(((center_pos.x - line_width) <= p_X) && (p_X < (center_pos.x + line_width))){
            rgb->x = line_color->x;
            rgb->y = line_color->y;
            rgb->z = line_color->z;
        }
    }
    return 0;
}


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


__DEVICE__ float2 draw_digits(int p_Width, int p_Height, int p_X, int p_Y, float3 *rgb, float drawing_value, float2 g_st_pos, int r_height, int r_width, float3 *font_color)
{
    int ii;
    int text_width = r_width + r_height * 4;
    int text_width_period = r_height * 4;
    float magnitude_value;
    int digit;
    int integer_digits;
    int decimal_digits;
    int drawing_value_int;
    integer_digits = calc_integer_digits(drawing_value);
    decimal_digits = TEXT_EFFECTIVE_DIGIT - integer_digits;

    // convert float to int for roundup.
    drawing_value_int = int(_round(drawing_value * _powf(10, decimal_digits)));

    // draw integer value
    float2 st_pos = g_st_pos;
    st_pos.x -= r_width;  // To neutralize the effect of the initial offset calculation in the for loop.
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

    st_pos.x += text_width;  // This is the margen between R, G, B.

    return st_pos;
}


__DEVICE__ int draw_rgb_digits(int p_Width, int p_Height, int p_X, int p_Y, float3 *rgb, float3 *drawing_rgb_value, float2 g_st_pos, int r_height, int r_width, float3 *font_color)
{
    float2 st_pos = g_st_pos;

    st_pos = draw_digits(p_Width, p_Height, p_X, p_Y, rgb, drawing_rgb_value->x, st_pos, r_height, r_width, font_color);
    st_pos = draw_digits(p_Width, p_Height, p_X, p_Y, rgb, drawing_rgb_value->y, st_pos, r_height, r_width, font_color);
    st_pos = draw_digits(p_Width, p_Height, p_X, p_Y, rgb, drawing_rgb_value->z, st_pos, r_height, r_width, font_color);
    return 0;
}


#endif
