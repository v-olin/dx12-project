struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float4 color : COLOR;
};

cbuffer ConstantBuffer : register(b0)
{
    float4x4 wvpMat;
    int pointLightCount;
};

float4 main(VS_OUTPUT input) : SV_TARGET
{
    if (pointLightCount > 0)
    {
        return float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    
    // return interpolated color
    return input.color;
}