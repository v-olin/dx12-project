#include "Common.hlsl"

// Raytracing output texture, accessed as a UAV
RWTexture2D< float4 > gOutput : register(u0);

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure SceneBVH : register(t0);

cbuffer CameraBuffer : register(b0)
{
    float4x4 view;
    float4x4 proj;
    float4x4 viewInv;
    float4x4 projInv;
}

[shader("raygeneration")] 
void RayGen() {
    HitInfo payload;
    payload.colorAndDistance = float4(0, 0, 0, 0);
    
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims = float2(DispatchRaysDimensions().xy);
    float2 d = (((launchIndex.xy + 0.5f) / dims.xy) * 2.f - 1.f); // pixel midpoint
    
    RayDesc ray;
    ray.Origin = mul(viewInv, float4(0, 0, 0, 1)).xyz;
    float4 target = mul(projInv, float4(d.x, -d.y, 1, 1));
    ray.Direction = mul(viewInv, float4(target.xyz, 0)).xyz;
    ray.TMin = 0;
    ray.TMax = 100000;
    
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
    
    if (payload.colorAndDistance.a < 0)
    {
        payload.colorAndDistance.rgb = float3(0.1, 0.1, 0.1);
    }

    gOutput[launchIndex] = float4(payload.colorAndDistance.rgb, 1.f);
}
