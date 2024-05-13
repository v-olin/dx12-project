struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
};

struct PointLight
{
    float4 position;
};

cbuffer ConstantBuffer : register(b0)
{
    float4x4 wvpMat; // 64 bytes
    float4x4 normalMatrix;
    float4x4 modelViewMatrix;
    float4x4 viewInverse;
    float4x4 viewMat;
    
    PointLight pointLights[3]; // 48 bytes
    int pointLightCount; // 4 bytes

    bool isProcWorld;
};

// Output color is white
float4 main(VS_OUTPUT input) : SV_TARGET
{
    // Output color is red
    return float4(1.0f, 0.0f, 0.0f, 1.0f);
}