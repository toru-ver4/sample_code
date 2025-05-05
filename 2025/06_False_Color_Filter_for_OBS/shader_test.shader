float LinearToSrgbChannel(float u)
{
    float s = sign(u);
    float abs_u = abs(u);
    return s * ((abs_u <= 0.0031308) ? (12.92 * abs_u) : ((1.055 * pow(abs_u, 1.0/2.4)) - 0.055));
}


float3 LinearToSRGB(float3 c)
{
    return float3(
        LinearToSrgbChannel(c.r),
        LinearToSrgbChannel(c.g),
        LinearToSrgbChannel(c.b)
    );
}

float4 mainImage(VertData v_in) : TARGET
{
	float4 rgba =image.Sample(textureSampler, v_in.uv);
    // if ((rgba.r < 10) & (rgba.g < 10) & (rgba.b < 10)) {
    //     // Red color detected, apply a false color effect
    //     rgba.r = 0.5; // Set red to full intensity
    //     rgba.g = 0.5; // Set green to zero
    //     rgba.b = 0.1; // Set blue to zero
    // }

    // rgba.rgb = LinearToSRGB(rgba.rgb);
	return rgba;
}
