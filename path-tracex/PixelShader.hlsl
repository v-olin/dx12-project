float4 main(float3 color : Color) : SV_TARGET
{
	//return float4(1.0f, 1.0f, 1.0f, 1.0f);
    return float4(color, 1.0f);
}