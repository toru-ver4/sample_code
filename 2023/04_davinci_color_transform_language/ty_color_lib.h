// Basic Color Library made by Toru Yoshihara
#ifndef TY_COLOR_LIB_H
#define TY_COLOR_LIB_H

#include "ampas_color_lib.h"

#define GAMUT_IDX_BT709  (0)
#define GAMUT_IDX_P3D65  (1)
#define GAMUT_IDX_BT2020 (2)

__CONSTANT__ float rgb2y_coef_bt709[] = {0.21263901f, 0.71516868f, 0.07219232f};
__CONSTANT__ float rgb2y_coef_p3d65[] = {0.22897456f, 0.69173852f, 0.07928691f};
__CONSTANT__ float rgb2y_coef_bt2020[] = {0.26270021f, 0.67799807f, 0.05930171f};

__CONSTANT__ float srgb_to_bt709_mtx[3][3] = {
    { 1.000000, 0.000000, 0.0000000 },
    { 0.000000, 1.000000, 0.0000000 },
    { -0.000000, -0.000000, 1.0000000 },
};
__CONSTANT__ float srgb_to_p3d65_mtx[3][3] = {
    { 0.822462, 0.177538, -0.0000000 },
    { 0.033194, 0.966806, -0.0000000 },
    { 0.017083, 0.072397, 0.9105199 },
};
__CONSTANT__ float srgb_to_bt2020_mtx[3][3] = {
    { 0.627404, 0.329283, 0.0433131 },
    { 0.069097, 0.919540, 0.0113623 },
    { 0.016391, 0.088013, 0.8955953 },
};


// Apply 3x3 matrix
__DEVICE__ float3 apply_matrix(float3 in, float mtx[3][3])
{
    float3 out;

    out.x = in.x * mtx[0][0] + in.y * mtx[0][1] + in.z * mtx[0][2];
    out.y = in.x * mtx[1][0] + in.y * mtx[1][1] + in.z * mtx[1][2];
    out.z = in.x * mtx[2][0] + in.y * mtx[2][1] + in.z * mtx[2][2];

    return out;
}


// Converts RGB to Y.
__DEVICE__ float rgb_2_y(float3 in, int gamut_idx)
{
    float y;
    float *coef;
    if(gamut_idx == GAMUT_IDX_BT709){
        coef = rgb2y_coef_bt709;
    }
    else if(gamut_idx == GAMUT_IDX_P3D65){
        coef = rgb2y_coef_p3d65;
    }
    else if(gamut_idx == GAMUT_IDX_BT2020){
        coef = rgb2y_coef_bt2020;
    }
    else{
        coef = rgb2y_coef_bt709;
    }
    y = coef[0] * in.x + coef[1] * in.y + coef[2] * in.z;

    return y;
}


// Gamma EOTF
__DEVICE__ float eotf_gamma(float in, float gamma)
{
    float y = _powf(in, gamma);
    return y;
}


__DEVICE__ float oetf_gamma(float in, float gamma)
{
    float y = _powf(in, 1.0f/gamma);
    return y;
}


__DEVICE__ float3 eotf_gamma24_f3(float3 in)
{
    float3 out;
    out.x = eotf_gamma(in.x, 2.4f);
    out.y = eotf_gamma(in.y, 2.4f);
    out.z = eotf_gamma(in.z, 2.4f);
    return out;
}

__DEVICE__ float3 oetf_gamma24_f3(float3 in)
{
    float3 out;
    out.x = oetf_gamma(in.x, 2.4f);
    out.y = oetf_gamma(in.y, 2.4f);
    out.z = oetf_gamma(in.z, 2.4f);
    return out;
}


// sRGB EOTF
__DEVICE__ float eotf_srgb(float in)
{
    float y;
    if(in <= 0.04045){
        y = in / 12.92f;
    }
    else{
        y = _powf((in + 0.055) / 1.055, 2.4);
    }
    return y;
}

// sRGB OETF
__DEVICE__ float oetf_srgb(float in)
{
    float y;
    if(in <= 0.0031308f){
        y = in * 12.92f;
    }
    else{
        y = 1.055f * _powf(in, 1.0f / 2.4f) - 0.055f;
    }
    return y;
}

__DEVICE__ float3 eotf_srgb_f3(float3 in)
{
    float3 out;
    out.x = eotf_srgb(in.x);
    out.y = eotf_srgb(in.y);
    out.z = eotf_srgb(in.z);
    return out;
}

__DEVICE__ float3 oetf_srgb_f3(float3 in)
{
    float3 out;
    out.x = oetf_srgb(in.x);
    out.y = oetf_srgb(in.y);
    out.z = oetf_srgb(in.z);
    return out;
}


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
