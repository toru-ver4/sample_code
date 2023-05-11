// Basic Color Library made by Toru Yoshihara

#include "ampas_color_lib.h"

#define GAMUT_IDX_BT709  (0)
#define GAMUT_IDX_P3D65  (1)
#define GAMUT_IDX_BT2020 (2)

__CONSTANT__ float rgb2y_coef_bt709[] = {0.21263901f, 0.71516868f, 0.07219232f};
__CONSTANT__ float rgb2y_coef_p3d65[] = {0.22897456f, 0.69173852f, 0.07928691f};
__CONSTANT__ float rgb2y_coef_bt2020[] = {0.26270021f, 0.67799807f, 0.05930171f};


// Converts RGB to Y.
__DEVICE__ float3 rgb_2_y(float3 in, int gamut_idx)
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

    return make_float3(y, y, y);
}
