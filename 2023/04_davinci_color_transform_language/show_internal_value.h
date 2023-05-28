DEFINE_UI_PARAMS(h_center_pos, Picker H Pos, DCTLUI_SLIDER_FLOAT, 0.5, 0.0, 1.0, 0.001)
DEFINE_UI_PARAMS(v_center_pos, Picker V Pos, DCTLUI_SLIDER_FLOAT, 0.5, 0.0, 1.0, 0.001)
DEFINE_UI_PARAMS(h_info_pos, Debug Info H Pos, DCTLUI_SLIDER_FLOAT, 0.5, 0.0, 1.0, 0.001)
DEFINE_UI_PARAMS(v_info_pos, Debug Info V Pos, DCTLUI_SLIDER_FLOAT, 0.5, 0.0, 1.0, 0.001)
DEFINE_UI_PARAMS(font_size_rate, Font Size Ratio, DCTLUI_SLIDER_INT, 20, 1, 100, 1)

__CONSTANT__ float3 cross_hair_color = {1.0, 0.0, 1.0};
__CONSTANT__ float3 seven_seg_color = {0.5, 0.5, 0.5};
__CONSTANT__ int digit_to_mask[] = {0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F};
#define TEXT_PERIOD_MASK (0x80)

__DEVICE__ int draw_cross_hair(int p_Width, int p_Height, int p_X, int p_Y, float3 *rgb, float h_center_pos, float v_center_pos)
{
    float3 *line_color = &cross_hair_color;

    float cross_hair_rate = 0.025;
    int line_width = 3;
    int h_pos = int(h_center_pos * p_Width + 0.5);
    int h_pos_st = int(h_pos - p_Height * cross_hair_rate);
    int h_pos_ed = int(h_pos + p_Height * cross_hair_rate);
    int v_pos = int(v_center_pos * p_Height + 0.5);
    int v_pos_st = int(v_pos - p_Height * cross_hair_rate);
    int v_pos_ed = int(v_pos + p_Height * cross_hair_rate);

    // h-line
    if((h_pos_st <= p_X) && (p_X < h_pos_ed)){
        if(((v_pos - line_width) <= p_Y) && (p_Y < (v_pos + line_width))){
            rgb->x = line_color->x;
            rgb->y = line_color->y;
            rgb->z = line_color->z;
        }
    }

    if((v_pos_st <= p_Y) && (p_Y < v_pos_ed)){
        if(((h_pos - line_width) <= p_X) && (p_X < (h_pos + line_width))){
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
    int max_digits = 5;
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
    decimal_digits = 6 - integer_digits;

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


// main function
__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, __TEXTURE__ p_TexR, __TEXTURE__ p_TexG, __TEXTURE__ p_TexB)
{
    float3 out;
    float r = _tex2D(p_TexR, p_X, p_Y) * 0.5f;
    float g = _tex2D(p_TexG, p_X, p_Y) * 0.5f;
    float b = _tex2D(p_TexB, p_X, p_Y) * 0.5f;
    out = make_float3(r, g, b);

    draw_cross_hair(p_Width, p_Height, p_X, p_Y, &out, h_center_pos, v_center_pos);

    float rectangle_width_rate = 0.05;
    float rectangle_height_rate = 0.01;
    float2 st_pos;
    float2 ed_pos;
    st_pos.x = p_Width * h_info_pos;
    st_pos.y = p_Height * v_info_pos;
    ed_pos.x = st_pos.x + p_Height * rectangle_width_rate;
    ed_pos.y = st_pos.y + p_Height * rectangle_height_rate;

    // draw_rectangle(p_Width, p_Height, p_X, p_Y, &out, st_pos, ed_pos, &seven_seg_color);

    // float2 digit_st_pos = make_float2(0, 0);
    float3 font_color = make_float3(1.0, 1.0, 0.0);
    const float font_size_int_max = 100.0;  // It is the same value with max value of the `font_size_rate`
    int r_height = int(float(p_Height * font_size_rate) / (font_size_int_max * 60.0f) + 0.5f);
    int r_width = r_height * 5;
    int text_one_width = r_width + r_height * 4;
    int text_one_width_dot = r_height * 4;
    float3 drawing_rgb_value = make_float3(0.00001, 999999, 3.14159265);

    // draw_digits(p_Width, p_Height, p_X, p_Y, &out, 10000, decimal_digits_ui, st_pos, r_height, r_width, &font_color);
    draw_rgb_digits(p_Width, p_Height, p_X, p_Y, &out, &drawing_rgb_value, st_pos, r_height, r_width, &font_color);

    return out;
}
