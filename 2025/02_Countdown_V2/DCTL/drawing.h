#ifndef TY_DRAWING_H
#define TY_DRAWING_H

#include "custom_types.h"
#include "utils.h"

#define NUM_OF_OUTLINE_POS   (6)

/**
 * @brief Draw rectangle filled with specified colors
 * 
 * @param p_Width canvas width
 * @param p_Height canvas height
 * @param p_X current horizontal position
 * @param p_Y current vertical position
 * @param rgb_in input data
 * @param st_pos start position
 * @param ed_pos end position
 * @param fill_color fill color
 * @return rgb pixel data
 */
__DEVICE__ float3 draw_rectangle(
    int p_Width, int p_Height,
    int p_X, int p_Y,
    float3 rgb_in,
    int2 st_pos, int2 ed_pos, float3 fill_color)
{
    float3 rgb_out = rgb_in;

    if((st_pos.x <= p_X) && (p_X < ed_pos.x)){
        if((st_pos.y <= p_Y) && (p_Y < ed_pos.y)){
            rgb_out.x = fill_color.x;
            rgb_out.y = fill_color.y;
            rgb_out.z = fill_color.z;
        }
    }
    return rgb_out;
}

__DEVICE__ float4 draw_rectangle_with_alpha(
    int p_Width, int p_Height,
    int p_X, int p_Y,
    float4 rgba_in,
    int2 st_pos, int2 ed_pos, float3 fill_color)
{
    float4 rgba_out = rgba_in;

    if((st_pos.x <= p_X) && (p_X < ed_pos.x)){
        if((st_pos.y <= p_Y) && (p_Y < ed_pos.y)){
            rgba_out.x = fill_color.x;
            rgba_out.y = fill_color.y;
            rgba_out.z = fill_color.z;
            rgba_out.w = 1.0f;
        }
    }
    return rgba_out;
}

/**
 * @brief Draw rectangle filled with specified colors
 * 
 * @param p_Width canvas width
 * @param p_Height canvas height
 * @param p_X current horizontal position
 * @param p_Y current vertical position
 * @param rgb_in input data
 * @param inner_st_pos start position
 * @param inner_ed_pos end position
 * @param line_width line width
 * @param fill_color fill color
 * @return rgb pixel data
 */
__DEVICE__ float3 draw_rectangle_outline(
    int p_Width, int p_Height,
    int p_X, int p_Y,
    float3 rgb_in,
    int2 inner_st_pos, int2 inner_ed_pos, int line_width,
    float3 fill_color)
{
    float3 out_rgb = rgb_in;
    int2 st_pos_list[NUM_OF_OUTLINE_POS] = {
        {inner_st_pos.x - line_width, inner_st_pos.y - line_width},
        {inner_st_pos.x - line_width, inner_ed_pos.y},
        {inner_st_pos.x - line_width, inner_st_pos.y - line_width},
        {inner_ed_pos.x, inner_st_pos.y - line_width}
    };
    int2 ed_pos_list[NUM_OF_OUTLINE_POS] = {
        {inner_ed_pos.x + line_width, inner_st_pos.y},
        {inner_ed_pos.x + line_width, inner_ed_pos.y + line_width},
        {inner_st_pos.x, inner_ed_pos.y + line_width},
        {inner_ed_pos.x + line_width, inner_ed_pos.y + line_width}
    };

    for(int ii=0; ii<NUM_OF_OUTLINE_POS; ++ii){
        out_rgb = draw_rectangle(
            p_Width, p_Height,
            p_X, p_Y,
            out_rgb,
            st_pos_list[ii], ed_pos_list[ii], fill_color);
    }
    return out_rgb;
}

__DEVICE__ float4 draw_rectangle_outline_with_alpha(
    int p_Width, int p_Height,
    int p_X, int p_Y,
    float4 rgba_in,
    int2 inner_st_pos, int2 inner_ed_pos, int line_width,
    float3 fill_color)
{
    float4 out_rgba = rgba_in;
    int2 st_pos_list[NUM_OF_OUTLINE_POS] = {
        {inner_st_pos.x - line_width, inner_st_pos.y - line_width},
        {inner_st_pos.x - line_width, inner_ed_pos.y},
        {inner_st_pos.x - line_width, inner_st_pos.y - line_width},
        {inner_ed_pos.x, inner_st_pos.y - line_width}
    };
    int2 ed_pos_list[NUM_OF_OUTLINE_POS] = {
        {inner_ed_pos.x + line_width, inner_st_pos.y},
        {inner_ed_pos.x + line_width, inner_ed_pos.y + line_width},
        {inner_st_pos.x, inner_ed_pos.y + line_width},
        {inner_ed_pos.x + line_width, inner_ed_pos.y + line_width}
    };

    for(int ii=0; ii<NUM_OF_OUTLINE_POS; ++ii){
        out_rgba = draw_rectangle_with_alpha(
            p_Width, p_Height,
            p_X, p_Y,
            out_rgba,
            st_pos_list[ii], ed_pos_list[ii], fill_color);
    }
    return out_rgba;
}


/**
 * @brief Draw cross 45degree line
 * 
 * @param p_Width canvas width
 * @param p_Height canvas height
 * @param p_X current horizontal position
 * @param p_Y current vertical position
 * @param rgb_in input data
 * @param offset horizontal offset from center position
 * @param v_center center position w/o offset
 * @param h_center center position w/o offset
 * @param line_width line width
 * @param fill_color fill color
 * @return rgb pixel data
 */
__DEVICE__ float3 draw_45deg_line(
    int p_Width, int p_Height,
    int p_X, int p_Y,
    float3 rgb_in,
    int offset, int h_center, int v_center,
    int line_width, float3 fill_color)
{
    float3 rgb_out = rgb_in;
    int local_offset;
    int h_pos_1;
    int h_pos_2;

    for(int ii=0; ii<line_width; ++ii){
        local_offset = -line_width/2 + ii;
        h_pos_1 = h_center - (p_Y - v_center) + offset + local_offset;
        h_pos_2 = h_center + (p_Y - v_center) + offset + local_offset;
        if(p_X == h_pos_1){
            rgb_out.x = fill_color.x;
            rgb_out.y = fill_color.y;
            rgb_out.z = fill_color.z;
        }
        if(p_X == h_pos_2){
            rgb_out.x = fill_color.x;
            rgb_out.y = fill_color.y;
            rgb_out.z = fill_color.z;
        }
    }

    return rgb_out;
}

#endif
