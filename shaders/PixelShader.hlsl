Texture2D colTex : register(t0);
Texture2D normalTex : register(t1);
Texture2D shinyTex : register(t2);
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
    bool hasColTex; // 1 bytes
    bool hasNormalTex;
    bool hasShinyTex;
    float material_shininess;
    float material_metalness;
    float material_fresnel;
  };

float3 calculatePointLight(PointLight light, float3 normal, float3 position, float3 baseColor)
{
    float3 lightDir = light.position.xyz - position;
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
float3 point_light_color = float3(1.0, 1.0, 1.0);
float point_light_intensity_multiplier = 50.0;
#define PI 3.14159265359

float3 calculateDirectIllumiunation(float3 wo, float3 n, float3 baseCol, VS_OUTPUT input)
{
    float direct_illum = float3(0, 0, 0);
    //for (int i = 0; i < pointLightCount; i++)
    //{
        
    //    PointLight light = pointLights[i];
    //    float3 direct_illum = baseCol;
	
    //    //float d = distance(viewSpaceLightPosition, viewSpacePosition);
    //    float d = distance(light.position, input.pos);

    //    float3 Li = point_light_intensity_multiplier * point_light_color * (1 / (d * d));

    //    float3 wi = normalize(light.position - input.pos);
	
    //    float dotp = dot(n, wi);

    //    if (dotp <= 0)
    //        return float3(0.0f, 0.0, 0.0);

    //    float3 diffuseTerm = baseCol * 1 / PI * abs(dotp) * Li;

    //    float3 wh = normalize(wi + wo); // can be too short
    //    float ndotwh = max(0.0001f, dot(n, wh));
    //    float ndotwo = max(0.0001f, dot(n, wo));
    //    float wodotwh = max(0.0001f, dot(wo, wh));
    //    float ndotwi = max(0.0001f, dot(n, wi));
    //    float whdotwi = max(0.0001f, dot(wh, wi));
        
    //    float F = material_fresnel + (1 - material_fresnel) * pow(1 - whdotwi, 5);
    //    float D = ((material_shininess + 2) / (2 * PI)) * pow(ndotwh, material_shininess);
        
    //    float left = 2 * (ndotwh * ndotwo) / wodotwh;
    //    float right = 2 * (ndotwh * ndotwi) / wodotwh;
    //    float G = min(1, min(left, right));

    //    float brdf = (F * D * G) / (4 * ndotwo * ndotwi);

    //    float3 dielectric_term = brdf * ndotwi * Li + (1 - F) * diffuseTerm;

    //    float3 metal_term = brdf * baseCol * ndotwi * Li;

    //    direct_illum += (material_metalness * metal_term + (1 - material_metalness) * dielectric_term) / pointLightCount;
    //}

    return direct_illum;
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
    //float3 wo = -normalize(input.pos);
    //float3 n = normalize(input.no);
    //result = calculateDirectIllumiunation()

    float4 color = float4(result, 1.0f);
    // return interpolated color
    ////also sample texture if it exists, needs to be implemented and passed in
    if (hasColTex)
        return colTex.Sample(s1, input.texCoord);
    return input.color;
}
