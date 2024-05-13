struct VS_INPUT
{
    float3 pos : POSITION;
};

struct PointLight
{
    float4 position;
};

struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
};

// Constant buffer structure for bounding box vertices
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

VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;
    
    // Transform vertex position from world space to homogeneous clip space
    output.pos = mul(float4(input.pos, 1.0f), wvpMat);

    return output;
}