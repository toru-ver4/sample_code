#ifndef TY_COLOR_PROCESSING_H
#define TY_COLOR_PROCESSING_H

#include "ampas_color_lib.h"

__DEVICE__ inline float3 oetf_st2084_f3(float3 in)
{
    return Linear_2_ST2084_f3(in);
}

__DEVICE__ inline float3 eotf_st2084_f3(float3 in)
{
    return ST2084_2_Linear_f3(in);
}

__DEVICE__ inline float oetf_st2084(float in)
{
    return Linear_2_ST2084(in);
}

__DEVICE__ inline float eotf_st2084(float in)
{
    return ST2084_2_Linear(in);
}

#endif
