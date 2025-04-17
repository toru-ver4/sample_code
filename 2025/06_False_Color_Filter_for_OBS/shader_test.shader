float4 mainImage(VertData v_in) : TARGET
{
	float4 rgba =image.Sample(textureSampler, v_in.uv);
    // if (rgba.r > 10.0) {
    //     // Red color detected, apply a false color effect
    //     rgba.r = 0.1; // Set red to full intensity
    //     rgba.g = 0.1; // Set green to zero
    //     rgba.b = 0.1; // Set blue to zero
    // }
	return rgba;
}
