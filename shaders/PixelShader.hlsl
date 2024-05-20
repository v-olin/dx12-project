Texture2D colTex : register(t0);
Texture2D colTex2 : register(t1);
Texture2D normalTex : register(t2);
Texture2D normalTex2 : register(t3);
Texture2D shinyTex : register(t4);
Texture2D metalTex : register(t5);
Texture2D fresnelTex : register(t6);
Texture2D emisionTex : register(t7);

SamplerState s1 : register(s0);
struct VS_OUTPUT
{
    float4 pos: SV_POSITION;
    float4 color : COLOR;
    float2 texCoord : TEXCOORD;
    //float4 worldNormal : WORLDNORMAL;
    //float4 worldPos : WORLDPOS;
    float4 viewSpaceNormal : VIEWSPACENORMAL;
    float4 viewSpacePos : VIEWSPACEPOS;
    float3 tangent :TANGENT;
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
cbuffer ConstantMeshBuffer : register(b1)
{
    float4 material_emmision;
    float4 material_color;
    bool hasColTex; 
    bool hasNormalTex;
    bool hasShinyTex;
    bool hasMetalTex;
    bool hasFresnelTex;
    bool hasEmisionTex;
    float material_shininess;
    float material_metalness;
    float material_fresnel;
    bool hasMaterial;

    
    //for proc world only, should maybe not be here...
	float stop_flat;
	float stop_interp;
    
}


#define PI 3.14159265359f




float3 calculateDirectIllumiunation(PointLight light, float3 wo, float3 n, float3 base_color, VS_OUTPUT input)
{
    float3 direct_illum = base_color;
    float shininess = material_shininess; 
    if(hasShinyTex)
    {
        float4 shiny = shinyTex.Sample(s1, input.texCoord);
        shininess = shiny.r;
    }
    float metalness = material_metalness;
    if (hasMetalTex)
    {
        float4 metal = metalTex.Sample(s1, input.texCoord);
        metalness = metal.r;
    }
	///////////////////////////////////////////////////////////////////////////
	// Task 1.2 - Calculate the radiance Li from the light, and the direction
	//            to the light. If the light is backfacing the triangle,
	//            return float3(0);
	///////////////////////////////////////////////////////////////////////////

    float3 point_light_color = float3(1.0, 1.0, 1.0);
    float point_light_intensity_multiplier = 500.0;

    
    float3 viewSpaceLightPosition = mul(float4(light.position.xyz, 1.0), viewMat).xyz;
    const float d = length(viewSpaceLightPosition - input.viewSpacePos.xyz);
    float3 Li = point_light_intensity_multiplier * point_light_color / (d * d);
    float3 wi = normalize(viewSpaceLightPosition - input.viewSpacePos.xyz);
    float ndotwi = dot(n, wi);
	
    if (ndotwi <= 0)
    {
        return float3(0.0, 0.0, 0.0);
    }

	///////////////////////////////////////////////////////////////////////////
	// Task 1.3 - Calculate the diffuse term and return that as the result
	///////////////////////////////////////////////////////////////////////////

    float3 diffuse_term = base_color * (1.0 / PI) * ndotwi * Li;
    direct_illum = diffuse_term;

	///////////////////////////////////////////////////////////////////////////
	// Task 2 - Calculate the Torrance Sparrow BRDF and return the light
	//          reflected from that instead
	///////////////////////////////////////////////////////////////////////////
    float3 wh = normalize(wi + wo);
    float ndotwh = max(0.0001, dot(n, wh));
    float ndotwo = max(0.0001, dot(n, wo));
    float wodotwh = max(0.0001, dot(wo, wh));
    float D = ((shininess + 2) / (2.0 * PI)) * pow(ndotwh, shininess);
    float G = min(1.0, min(2.0 * ndotwh * ndotwo / wodotwh, 2.0 * ndotwh * ndotwi / wodotwh));
    float F = material_fresnel + (1.0 - material_fresnel) * pow(1.0 - wodotwh, 5.0);
    float denominator = 4.0 * clamp(ndotwo * ndotwi, 0.0001, 1.0);
    float brdf = D * F * G / denominator;



	///////////////////////////////////////////////////////////////////////////
	// Task 3 - Make your shader respect the parameters of our material model.
	///////////////////////////////////////////////////////////////////////////
    float3 dielectric_term = brdf * ndotwi * Li + (1 - F) * diffuse_term;
    float3 metal_term = brdf * base_color * ndotwi * Li;
    direct_illum = metalness * metal_term + (1 - metalness) * dielectric_term;


    return direct_illum;
}
float3 calculateIndirectIllumination(float3 wo, float3 n, float3 base_color, VS_OUTPUT input)
{
    float3 indirect_illum = float3(0.0, 0.0, 0.0);
	///////////////////////////////////////////////////////////////////////////
	// Task 5 - Lookup the irradiance from the irradiance map and calculate
	//          the diffuse reflection
	///////////////////////////////////////////////////////////////////////////
    float3 world_normal = mul(float4(n, 1.0), viewInverse).xyz;
	// Calculate the spherical coordinates of the direction
    float theta = acos(max(-1.0f, min(1.0f, world_normal.y)));
    float phi = atan(world_normal.z / world_normal.x);
    if (phi < 0.0f)
        phi = phi + 2.0f * PI;

	// Use these to lookup the color in the environment map
    float2 lookup = float2(phi / (2.0 * PI), 1 - theta / PI);
    

    //float3 Li = environment_multiplier * texture(irradianceMap, lookup).rgb;
    float3 env_irradiance = (1, 1, 1);
    float Li = env_irradiance;

    float3 diffuse_term = base_color * (1.0 / PI) * Li;
	
    indirect_illum = diffuse_term;
    return indirect_illum;

}
float3 calcProcWorldCol(VS_OUTPUT input)
{
    float3 worldNormal = mul(float4(input.viewSpaceNormal.xyz, 0.0), viewInverse);
    float3 worldpos = mul(float4(input.viewSpacePos.xyz, 1.0), viewInverse);
    float slope = dot(normalize(worldNormal), float3(0, 1, 0)); // Cosine of the angle between the normal and the up vector

    float3 flat_col = colTex.Sample(s1, input.texCoord);
    float3 slope_col = colTex2.Sample(s1, input.texCoord); // 
    float3 interpolatedColor;

    if(slope > stop_flat) //flat
        interpolatedColor = flat_col;
    else if(slope < stop_interp) //steep
        interpolatedColor = slope_col;
    else //interp
    {
        //remap slope from [stop_interpet, stop_flat] to [1.0]
        float remapped_slope = saturate((slope - stop_interp) / (stop_flat - stop_interp));
        interpolatedColor = lerp(slope_col, flat_col, remapped_slope);
    }
    return interpolatedColor;
}
float4 main(VS_OUTPUT input) : SV_TARGET
{
    if (pointLightCount == 0)
    {
        return input.color;
    }
    float3 color = input.color.rgb;
    
    if (hasMaterial)
        color = material_color.rgb;

    bool invalidTexCoords = abs(input.texCoord.x + 1) < 0.01 && abs(input.texCoord.y + 1) < 0.01;
    
    if (hasColTex && !invalidTexCoords)
        color = colTex.Sample(s1, input.texCoord);
    
    if (isProcWorld)
        color = calcProcWorldCol(input);
    
    float3 result = float3(0, 0, 0);

    //float3 normal = input.worldNormal.xyz;

    float3 viewSpaceNormal = input.viewSpaceNormal;
    float3 normal = input.viewSpaceNormal.xyz;
    if (hasNormalTex && !invalidTexCoords)
    {
        float4 normalMap = normalTex.Sample(s1, input.texCoord);
        //Change normal map range from [0, 1] to [-1, 1]
        normalMap = (2.0f * normalMap) - 1.0f;
        //Make sure tangent is completely orthogonal to normal
        input.tangent = normalize(input.tangent - dot(input.tangent, normal) * normal);
        //Create the biTangent
        float3 biTangent = cross(normal, input.tangent);
        //Create the "Texture Space"
        float3x3 texSpace = float3x3(input.tangent, biTangent, normal);
        //Convert normal from normal map to texture space and store in input.normal
        float3 modelNormal = normalize(mul(normalMap.xyz, texSpace));
        viewSpaceNormal = modelNormal;
    }
       
    float3 wo = -normalize(input.viewSpacePos.xyz);
    float3 n = normalize(viewSpaceNormal);
        
    for (int i = 0; i < pointLightCount; i++)
    {
        result += calculateDirectIllumiunation(pointLights[i], wo, n, color, input);
    }
    result += calculateIndirectIllumination(wo, n, color, input);


    float3 emision = material_emmision.rgb;
    if(hasEmisionTex)
        emision = emisionTex.Sample(s1, input.texCoord);

    result += emision;
    return float4(result, 1.0f);
}