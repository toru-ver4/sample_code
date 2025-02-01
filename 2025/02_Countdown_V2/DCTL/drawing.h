#ifndef TY_DRAWING_H
#define TY_DRAWING_H

#include "custom_types.h"

#define NUM_OF_OUTLINE_POS   (6)

/**
 * @brief Draw rectangle filled with specified colors
 * 
 * @param p_Width canvas width
 * @param p_Height canvas height
 * @param p_x current horizontal position
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


/**
 * @brief Draw rectangle filled with specified colors
 * 
 * @param p_Width canvas width
 * @param p_Height canvas height
 * @param p_x current horizontal position
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

    for(int ii=0; ii<NUM_OF_OUTLINE_POS; ii++){
        out_rgb = draw_rectangle(
            p_Width, p_Height,
            p_X, p_Y,
            out_rgb,
            st_pos_list[ii], ed_pos_list[ii], fill_color);
    }
    return out_rgb;
}

#endif
