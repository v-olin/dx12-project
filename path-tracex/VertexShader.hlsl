struct VSOut
{
	// order is important!!
	// order must match argument order in main of PixelShader
    float3 color : Color;
    float4 pos : SV_Position;
};

/*
cbuffer Cbuf
{
    matrix transform;
}
*/

// <type> name : <semantic name>
VSOut main(float3 pos : Position, float3 color : Color)
{
    VSOut vso;
    vso.pos = float4(pos, 1.0f);
    vso.color = color;
    return vso;
}