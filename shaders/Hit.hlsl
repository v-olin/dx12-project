#include "Common.hlsl"

struct ShadowHitInfo
{
    bool isHit;
};

struct PointLight
{
    float4 position;
    float4 color;
    float intensity;
};

struct MeshData
{
    float4 material_color;
    float4 material_emmision;
    float4x4 normal_matrix;
    bool hasColTex;
    bool hasNormalTex;
    bool hasShinyTex;
    bool hasMetalTex;
    bool hasFresnelTex;
    bool hasEmisionTex;
    float material_shininess;
    float material_metalness;
    float material_fresnel;
    float material_transparency;
    bool hasMaterial;
};

cbuffer CameraBuffer : register(b0)
{
    float4x4 view;
    float4x4 proj;
    float4x4 viewInv;
    float4x4 projInv;
}

RWTexture2D<float4> noiseTex : register(u0);
StructuredBuffer<Vertex> Vertices : register(t0);
StructuredBuffer<int> indices : register(t1);
RaytracingAccelerationStructure SceneBVH : register(t2);
StructuredBuffer<MeshData> meshdatas : register(t3);

cbuffer LightBuffer : register(b1)
{
    PointLight lights[5];
    int lightCount;
}

// Retrieve attribute at a hit position interpolated from vertex attributes using the hit's barycentrics.
float3 HitAttribute(float3 vertexAttribute[3], Attributes attr)
{
    return vertexAttribute[0] +
        attr.bary.x * (vertexAttribute[1] - vertexAttribute[0]) +
        attr.bary.y * (vertexAttribute[2] - vertexAttribute[0]);
}

// Retrieve hit world position.
float3 HitWorldPosition()
{
    return WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
}

float random(float2 st)
{
    return frac(sin(dot(st.xy, float2(12.9898, 78.233))) * 43758.5453123);
}

float ambientOcclusion(float3 normal, float3 position, float3 rayDir, float2 seed)
{
    float ao = 0.0f;
    float3 p = position + 0.001f * normal;
    int rayCount = 10;

    for (int i = 0; i < rayCount; i++)
    {
        // TODO: fix randomness for the direction
        //float realSeed = random(seed); // This is shit TODO: FIX
        //float realSeed2 = random(float2(DispatchRaysIndex().x ^ 50, DispatchRaysIndex().y ^ 50)); // This is shit TODO: FIX
        //float realSeed3 = random(seed + float2(DispatchRaysIndex().x, DispatchRaysIndex().y)); // This is shit TODO: FIX
        // Shoot ray in random direction towards hemisphere
        
        float xp = noiseTex[DispatchRaysIndex().xy].r;
        float yp = noiseTex[DispatchRaysIndex().yx].r;
        
        uint2 test = uint2(xp * 1280, yp * 720);
        float rand = noiseTex[test].r;
        float3 r = normalize(float3(rand, 1, 0));
        
        if (dot(normal, r) < 0.0f)
        {
            r = -r;
        }
        
        RayDesc ray;
        ray.Origin = position + 0.001f * r;
        ray.Direction = r;
        ray.TMin = 0.01f;
        
        
        ray.TMax = 0.1f; // This parameter will change how much darkness is present in the corners
        
        ShadowHitInfo shadowPayload;
        shadowPayload.isHit = false;
        
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
        
        if (shadowPayload.isHit)
        {
            ao += 1.0f;
        }
        
    }
    return 1.0f - ao / rayCount;
}

float3 GetMaterialColor(Attributes attrib)
{
    uint vertId = 3 * PrimitiveIndex();
    uint matidx = Vertices[vertId].materialIdx;
    
    if (meshdatas[matidx].hasMaterial)
    {
        float3 colors[3] =
        {
            meshdatas[Vertices[vertId + 0].materialIdx].material_color.rgb,
            meshdatas[Vertices[vertId + 1].materialIdx].material_color.rgb,
            meshdatas[Vertices[vertId + 2].materialIdx].material_color.rgb
        };
        
        float3 hitColor = HitAttribute(colors, attrib);
        return hitColor;
    }
    else
    {
        float3 colors[3] = 
        {
            Vertices[indices[vertId + 0]].color.rgb,
            Vertices[indices[vertId + 1]].color.rgb,
            Vertices[indices[vertId + 2]].color.rgb
        };
        
        // blend them with bary
        float3 hitColor = HitAttribute(colors, attrib);
        return hitColor;
    }
}

float3 calcuateReflectionRayContribution(float3 normal, float3 viewDir)
{
        // Reflection ray
    float3 reflectDir = reflect(viewDir, normal);
    RayDesc reflectRay;
    reflectRay.Origin = HitWorldPosition() + 0.001f * reflectDir;
    reflectRay.Direction = reflectDir;
    reflectRay.TMin = 0.01f;
    reflectRay.TMax = 10000.f;
    
        // Trace reflection ray
    HitInfo reflectPayload;
    reflectPayload.colorAndDistance = float4(0, 0, 0, -100);
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, reflectRay, reflectPayload);

    float3 outColor = reflectPayload.colorAndDistance.xyz;

    if (reflectPayload.colorAndDistance.w == -1)
    {
        outColor = float3(0, 0, 0);
    }
        
    return outColor;
}

float3 calculateTransparantRayContribution(float3 normal, float3 viewDir)
{
    // Shoot a transmission ray
    float3 transmitDir = viewDir;
        
    float3 refractDir = refract(viewDir, normal, 1.125);
    RayDesc refractRay;
    refractRay.Origin = HitWorldPosition() + 0.1f * transmitDir;
    refractRay.Direction = transmitDir;
    refractRay.TMin = 0.01f;
    refractRay.TMax = 10000.f;
        
        
        // Trace transmission ray
    HitInfo refractPayload;
    refractPayload.colorAndDistance = float4(0, 0, 0, -101);
    TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 0, 0, refractRay, refractPayload);
        
    float3 outColor = refractPayload.colorAndDistance.xyz;
        
    if (refractPayload.colorAndDistance.w == -1)
    {
        outColor = float3(0, 0, 0);
    }
    
    return outColor;
}

[shader("closesthit")] void ClosestHit(inout HitInfo payload, Attributes attrib) 
{
    float RAY_LENGTH = RayTCurrent();
    uint vertId = 3 * PrimitiveIndex();
    
    // The color of the hit pixel without doing lighting calculations
    float3 hitColor = GetMaterialColor(attrib);
    
    // The color of the hit pixel after doing lighting calculations
    float3 outColor = float3(0, 0, 0);
    
    float shadowFactor = 0.0f;
    float3 worldOrigin = HitWorldPosition();
    float3 normals[3] = { Vertices[indices[vertId + 0]].normal, Vertices[indices[vertId + 1]].normal, Vertices[indices[vertId + 2]].normal };
    float3 normal = HitAttribute(normals, attrib);
    // Transform to world space
    normal = normalize(mul(float4(normalize(normal), 0.0f), meshdatas[Vertices[vertId].materialIdx].normal_matrix).xyz);
    
    // Dir from camera to hit point
    float3 viewDir = normalize(worldOrigin - WorldRayOrigin());
    
    // Only accept values between 0 and 1
    // 1 is fully reflective, 0 is fully diffuse
    float material_shininess = min(meshdatas[Vertices[vertId].materialIdx].material_shininess, 1);
    float material_transparency = min(meshdatas[Vertices[vertId].materialIdx].material_transparency, 1);
    
    float reflectiveContribution = (1 - material_transparency) * material_shininess;
    
    // In order to save ray payload memory we encode the type of ray in the w component
    // -100 is a reflection ray
    // -101 is a transmission ray
    if (payload.colorAndDistance.w != -100 && material_shininess > 0)
    {
        outColor += reflectiveContribution * calcuateReflectionRayContribution(normal, viewDir);
    }
    
    if (payload.colorAndDistance.w != -101 && material_transparency > 0)
    {
        outColor += material_transparency * calculateTransparantRayContribution(normal, viewDir);
    }
       
    for (int i = 0; i < lightCount; i++)
    {
        PointLight plight = lights[i];
        float3 lightPos = float3(plight.position.x, plight.position.y, plight.position.z);
        
        float3 lightDir = normalize(lightPos - worldOrigin);
        float lightDistance = length(lightPos - worldOrigin);
    
        RayDesc ray;
        ray.Origin = worldOrigin + 0.001f * lightDir;
        ray.Direction = lightDir;
        ray.TMin = 0.01f;
        ray.TMax = lightDistance;
        bool hit = true;
        
        ShadowHitInfo shadowPayload;
        shadowPayload.isHit = false;
                
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
        
        if (shadowPayload.isHit)
            continue;
        
        // Rendering equation
        float3 lightColor = float3(1, 1, 1);
        float diffuse = max(dot(normal, lightDir), 0.0f);
        
        // specular shading
        float3 halfVector = normalize(lightDir + viewDir);
        float3 vertexToCamera = normalize(WorldRayOrigin() - worldOrigin);
        float3 lightReflect = reflect(-lightDir, normal);
        float specular = pow(max(dot(vertexToCamera, lightReflect), 0.0f), 32);
        
        float directIlluminationContribution = max(1 - material_transparency - reflectiveContribution, 0);
        //float directIlluminationContribution = max(1 - material_transparency - 0.5, 0);
        outColor += directIlluminationContribution * hitColor * (diffuse * lightColor + specular * lightColor);
    }

    float3 ao = ambientOcclusion(normal, worldOrigin, WorldRayDirection(), attrib.bary);
    ao = outColor;
    payload.colorAndDistance = float4(ao.x, ao.y, ao.z, RAY_LENGTH);
    
    //uint2 testIdx = uint2(DispatchRaysIndex().x, DispatchRaysIndex().y);
    //float3 noise = noiseTex[testIdx].rgb;
    //payload.colorAndDistance = float4(noise.r, noise.g, noise.b, RayTCurrent());
    //payload.colorAndDistance = float4(outColor.x, outColor.y, outColor.z, RayTCurrent());
    //payload.colorAndDistance = float4(hitColor.x, hitColor.y, hitColor.z, RayTCurrent());
}

[shader("closesthit")] void PlaneClosestHit(inout HitInfo payload, Attributes attrib)
{
    float shadowFactor = 0.0f;
    
    for (int i = 0; i < lightCount && i < 1; i++)
    {
        PointLight plight = lights[i];
        float3 lightPos = float3(plight.position.x, plight.position.y, plight.position.z);
        
        float3 worldOrigin = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
        float3 lightDir = normalize(lightPos - worldOrigin);
        
        RayDesc ray;
        ray.Origin = worldOrigin;
        ray.Direction = lightDir;
        ray.TMin = 0.01f;
        ray.TMax = 100000;
        bool hit = true;
        
        ShadowHitInfo shadowPayload;
        shadowPayload.isHit = false;
        
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
        
        shadowFactor += shadowPayload.isHit ? 0.7f : 0.0f;
    }
    
    float3 color = float3(1, 1, 1) * (1 - shadowFactor);
    
    payload.colorAndDistance = float4(color.r, color.g, color.b, RayTCurrent());
}
