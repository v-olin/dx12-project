#include "Common.hlsl"

struct ShadowHitInfo
{
    bool isHit;
};

struct PointLight
{
    float4 position;
};

cbuffer CameraBuffer : register(b1)
{
    float4x4 view;
    float4x4 proj;
    float4x4 viewInv;
    float4x4 projInv;
}

StructuredBuffer<Vertex> Vertices : register(t0);
StructuredBuffer<int> indices : register(t1);
RaytracingAccelerationStructure SceneBVH : register(t2);

cbuffer LightBuffer : register(b0)
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
        float realSeed = random(seed);
        float realSeed2 = random(float2(DispatchRaysIndex().x ^ 50, DispatchRaysIndex().y ^ 50));
        float realSeed3 = random(seed + float2(DispatchRaysIndex().x ^ 70, DispatchRaysIndex().y ^ 80));
        // Shoot ray in random direction towards hemisphere
        float3 r = normalize(float3(random(realSeed2), random(realSeed), random(realSeed3)));
        
        if (dot(r, normal) < 0.0f)
        {
            r = -r;
        }
        
        RayDesc ray;
        ray.Origin = position;
        ray.Direction = r;
        ray.TMin = 0.01f;
        ray.TMax = 0.2f; //10000.f;
        
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

[shader("closesthit")] void ClosestHit(inout HitInfo payload, Attributes attrib) 
{
    uint vertId = 3 * PrimitiveIndex();
    float3 colors[3] = {Vertices[indices[vertId + 0]].color.rgb, Vertices[indices[vertId + 1]].color.rgb, Vertices[indices[vertId + 2]].color.rgb};
    float3 hitColor = HitAttribute(colors, attrib);
    
    float shadowFactor = 0.0f;
    float3 worldOrigin = HitWorldPosition();
    
    for (int i = 0; i < lightCount; i++)
    {
        PointLight plight = lights[i];
        float3 lightPos = float3(plight.position.x, plight.position.y, plight.position.z);
        
        float3 lightDir = normalize(lightPos - worldOrigin);
        float lightDistance = length(lightPos - worldOrigin);
        
        RayDesc ray;
        ray.Origin = worldOrigin;
        ray.Direction = lightDir;
        ray.TMin = 0.01f;
        ray.TMax = lightDistance; //10000.f;
        bool hit = true;
        
        ShadowHitInfo shadowPayload;
        shadowPayload.isHit = false;
        
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
        
        shadowFactor += shadowPayload.isHit ? (0.7f / float(lightCount)) : 0.0f;
    }
    
    hitColor = hitColor * (1.0f - shadowFactor);
  
    payload.colorAndDistance = float4(hitColor.x, hitColor.y, hitColor.z, RayTCurrent());

    float3 normals[3] = {Vertices[indices[vertId + 0]].normal, Vertices[indices[vertId + 1]].normal, Vertices[indices[vertId + 2]].normal};
    
    // TODO: transform the normal to world space, we need the model matrix
    float3 normal = mul(float4(normalize(HitAttribute(normals, attrib)), 1), normalMatrix);
    
    float2 st = float2(attrib.bary.x, attrib.bary.y);

    float ao = ambientOcclusion(normal, worldOrigin, WorldRayDirection(), st);
    
    payload.colorAndDistance = float4(hitColor, RayTCurrent());
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
