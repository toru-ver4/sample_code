#ifndef TY_UTILS_H
#define TY_UTILS_H

__DEVICE__ float3 rgba_to_rgb(float4 rgba_in)
{
    float3 rgb_out = {rgba_in.x, rgba_in.y, rgba_in.z};
    return rgb_out;
}

__DEVICE__ float4 rgb_to_rgba(float3 rgb_in)
{
    float4 rgba_out = {rgb_in.x, rgb_in.y, rgb_in.z, 1.0f};
    return rgba_out;
}

#endif
