Texture2D t1 : register(t0);
SamplerState s1 : register(s0);
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

float3 calculatePointLight(PointLight light, float3 normal, float3 position, float3 baseColor)
{
    float3 lightDir = light.position - position;
    float distance = length(lightDir);
    lightDir /= distance;
    
    float3 diffuse = baseColor * max(dot(lightDir, normal), 0.0f);
    
    // ambient light
    float3 ambient = baseColor * 0.2f;
    
    // specular light
    float3 viewDir = normalize(-position);
    float3 reflectDir = reflect(-lightDir, normal);
    float3 specular = baseColor * pow(max(dot(viewDir, reflectDir), 0.0f), 32.0f);
    
    // attenuation
    float attenuation = 1.0f / (1.0f + 0.1f * distance + 0.01f * distance * distance);
    
    diffuse *= attenuation;
//    ambient *= attenuation;
    specular *= attenuation;
    
    return diffuse + ambient + specular;
}

float4 main(VS_OUTPUT input) : SV_TARGET
{
    if (pointLightCount == 0)
    {
        return input.color;
    }
    
    float3 result = float3(0, 0, 0);
    for (int i = 0; i < pointLightCount; i++)
    {
        result += calculatePointLight(pointLights[i], input.worldNormal.xyz, input.worldPos.xyz, input.color);
    }

    float4 color = float4(result, 1.0f);
    // return interpolated color
    //also sample texture if it exists, needs to be implemented and passed in
    return t1.Sample(s1, input.texCoord);
   // return input.color;
}