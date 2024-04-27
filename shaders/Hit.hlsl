#include "Common.hlsl"

struct ShadowHitInfo
{
    bool isHit;
};

struct PointLight
{
    float4 position;
};

StructuredBuffer<Vertex> Vertices : register(t0);
StructuredBuffer<int> indices : register(t1);
RaytracingAccelerationStructure SceneBVH : register(t2);

cbuffer LightBuffer : register(b0)
{
    PointLight lights[5];
    int lightCount;
}

[shader("closesthit")] void ClosestHit(inout HitInfo payload, Attributes attrib) 
{
    float3 barycentrics = float3(1.f - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);
    uint vertId = 3 * PrimitiveIndex();
    float3 hitColor = Vertices[indices[vertId + 0]].color * barycentrics.x +
                      Vertices[indices[vertId + 1]].color * barycentrics.y +
                      Vertices[indices[vertId + 2]].color * barycentrics.z;
    
    float shadowFactor = 0.0f;
    
    for (int i = 0; i < lightCount; i++)
    {
        PointLight plight = lights[i];
        float3 lightPos = float3(plight.position.x, plight.position.y, plight.position.z);
        
        float3 worldOrigin = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
        float3 lightDir = normalize(lightPos - worldOrigin);
        
        RayDesc ray;
        ray.Origin = worldOrigin;
        ray.Direction = lightDir;
        ray.TMin = 0.01f;
        ray.TMax = 10000.f;
        bool hit = true;
        
        ShadowHitInfo shadowPayload;
        shadowPayload.isHit = false;
        
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
        
        shadowFactor += shadowPayload.isHit ? (0.7f / float(lightCount)) : 0.0f;
    }
    
    hitColor = hitColor * (1.0f - shadowFactor);
  
    payload.colorAndDistance = float4(hitColor.x, hitColor.y, hitColor.z, RayTCurrent());
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
