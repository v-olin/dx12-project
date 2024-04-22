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


#define PI 3.14159265359f



float3 calculateDirectIllumiunation(PointLight light, float3 normal, float3 position, float3 baseColor, float2 texCoord)
{
    float shininess = material_shininess;
    if(hasShinyTex)
        shininess = shinyTex.Sample(s1, texCoord).r;
    float metalness = material_metalness;
    float fresnel = material_fresnel;
    normal = normalize(normal);
    
    float3 point_light_color = float3(1.0, 1.0, 1.0);
    float point_light_intensity_multiplier = 50.0;

    float3 lightDir = light.position.xyz - position;
    float distance = length(lightDir);
    lightDir /= distance;


    float d = distance;
    float3 Li = point_light_intensity_multiplier * point_light_color; // dont care about light distance, for debug only * (1 / (d * d));

    float3 wi = lightDir;
	
    float dotp = dot(normal, wi);

    if (dotp <= 0)
    {
        return float3(0.0f, 0.0, 0.0);
    }

    float3 diffuse = baseColor * 1 / PI * max(abs(dotp), 0.0) * Li;
    float3 wo = -normalize(position);
    float3 wh = normalize(wi + wo); // can be too short
    float ndotwh = max(0.0001f, dot(normal, wh));
    float ndotwo = max(0.0001f, dot(normal, wo));
    float wodotwh = max(0.0001f, dot(wo, wh));
    float ndotwi = max(0.0001f, dot(normal, wi));
    float whdotwi = max(0.0001f, dot(wh, wi));
        
    float F = fresnel + (1 - fresnel) * pow(1 - whdotwi, 5);
    float D = ((shininess + 2) / (2 * PI)) * pow(ndotwh, shininess);
        
    float left = 2 * (ndotwh * ndotwo) / wodotwh;
    float right = 2 * (ndotwh * ndotwi) / wodotwh;
    float G = min(1, min(left, right));

    float brdf = (F * D * G) / (4 * ndotwo * ndotwi);

    float3 dielectric_term = brdf * ndotwi * Li + (1 - F) * diffuse;
    float3 metal_term = brdf * baseColor * ndotwi * Li;

    float3 direct_illum = (metalness * metal_term + (1 - metalness) * dielectric_term);
    
    return direct_illum;
  }

float4 main(VS_OUTPUT input) : SV_TARGET
{
    if (pointLightCount == 0)
    {
        return input.color;
    }
    float3 color = input.color.rgb;
    if (hasColTex)
        color = colTex.Sample(s1, input.texCoord);
    
    float3 result = float3(0, 0, 0);

    float3 normal = input.worldNormal.xyz;
    if (hasNormalTex)
    {

        normal = normalTex.Sample(s1, input.texCoord).xyz;
        normal = normalize(mul(float4(normal, 0), normalMatrix));
    }
       
    for (int i = 0; i < pointLightCount; i++)
    {
        result += calculateDirectIllumiunation(pointLights[i], normal, input.worldPos.xyz, color, input.texCoord);
    }

    return float4(result, 1.0f);
}