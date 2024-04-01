struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float4 color : COLOR;
};

struct PointLight
{
    float3 position;
};

cbuffer ConstantBuffer : register(b0)
{
    float4x4 wvpMat;
    int pointLightCount;
    PointLight pointLights[10];
};

float3 calculatePointLight(PointLight light, float3 normal, float3 position)
{
    float3 lightDir = light.position - position;
    float distance = length(lightDir);
    lightDir /= distance;
    
    float intensity = max(dot(normal, lightDir), 0.0f);
    
    return intensity;
}

float4 main(VS_OUTPUT input) : SV_TARGET
{
    if (pointLightCount > 0)
    {
        return float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    
    //float3 result = calculatePointLight(pointLights[0], float3(0.0f, 0.0f, 1.0f), float3(0.0f, 0.0f, 0.0f));
    
    // return interpolated color
    return input.color;
}