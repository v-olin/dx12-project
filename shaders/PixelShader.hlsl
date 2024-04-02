struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float4 color : COLOR;
    float2 texCoord : TEXCOORD;
    float4 worldNormal : WORLDNORMAL;
    float4 worldPos : WORLDPOS;
};

struct PointLight
{
    float4 position;
};

cbuffer ConstantBuffer : register(b0)
{
    float4x4 wvpMat; // 64 bytes
    float4x4 modelMatrix; // 64 bytes
    float4x4 normalMatrix; // 64 bytes
    PointLight pointLights[3]; // 48 bytes
    int pointLightCount; // 4 bytes
};

float3 calculatePointLight(PointLight light, float3 normal, float3 position)
{
    float3 lightDir = light.position - position;
    float distance = length(lightDir);
    lightDir /= distance;
    
    float intensity = max(dot(lightDir, normal), 0.0f);
    
    return intensity;
}

float4 main(VS_OUTPUT input) : SV_TARGET
{
    if (pointLightCount == 0)
    {
        return input.color;
    }
    
    float3 result = calculatePointLight(pointLights[0], input.worldNormal.xyz, input.worldPos.xyz);
    float4 color = float4(result, 1.0f);
    // return interpolated color
    return color;
}