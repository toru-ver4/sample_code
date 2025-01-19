#ifndef TY_DRAWING_H
#define TY_DRAWING_H

#include "custom_types.h"

/**
 * @brief Draw rectangle filled with specified colors
 * 
 * @param p_Width canvas width
 * @param p_Height canvas height
 * @param p_x current horizontal position
 * @param p_Y current vertical position
 * @param st_pos start position
 * @param ed_pos end position
 * @param fill_color fill color
 * @return 
 */
__DEVICE__ int draw_rectangle(
    int p_Width, int p_Height,
    int p_X, int p_Y,
    float3 *rgb,
    int2 st_pos, int2 ed_pos, float3 *fill_color)
{
    if((st_pos.x <= p_X) && (p_X < ed_pos.x)){
        if((st_pos.y <= p_Y) && (p_Y < ed_pos.y)){
            rgb->x = fill_color->x;
            rgb->y = fill_color->y;
            rgb->z = fill_color->z;
        }
    }

    return 0;
}

#endif
