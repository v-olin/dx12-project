#include "Common.hlsl"

// Raytracing output texture, accessed as a UAV
RWTexture2D<float4> gOutput : register(u0);
RWTexture2D<float4> gDepth : register(u1);

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure SceneBVH : register(t0);

cbuffer CameraBuffer : register(b0)
{
    float4x4 currView;
    float4x4 currProj;
    float4x4 currViewInv;
    float4x4 currProjInv;
    float4x4 prevView;
    float4x4 prevProj;
    float4x4 prevViewInv;
    float4x4 prevProjInv;
    float nearPlane;
    float farPlane;
    bool useTAA;
}

float getDepthValue(float zCoord)
{
    float depth = ((1 / zCoord) - (1 / nearPlane)) / ((1 / farPlane) - (1 / nearPlane));
    float linearDepth = (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - (depth * 2.0 - 1.0) * (farPlane - nearPlane));
    return linearDepth / farPlane;
}

[shader("raygeneration")] 
void RayGen() {
    PathHitInfo payload;
    payload.colorAndDistance = float4(0, 0, 0, 0);
    
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims = float2(DispatchRaysDimensions().xy);
    
    // this is the pixel midpoint we want to shoot the ray through
    float2 d = (((launchIndex.xy + 0.5f) / dims.xy) * 2.f - 1.f);
    
    RayDesc ray;
    
    // camera is its own origin
    ray.Origin = mul(currViewInv, float4(0, 0, 0, 1)).xyz;
    
    // target pixel projected to camera space
    float4 target = mul(currProjInv, float4(d.x, -d.y, 1, 1));
    ray.Direction = mul(currViewInv, float4(target.xyz, 0)).xyz;
    
    ray.TMin = nearPlane;
    ray.TMax = farPlane;
    
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    if (payload.colorAndDistance.a < 0)
    {
        payload.colorAndDistance.rgb = float3(0.1, 0.1, 0.1);
    } 
        
    gOutput[launchIndex] = float4(payload.colorAndDistance.rgb, 1.f);
    
    if (useTAA)
    {
        if (payload.colorAndDistance.a >= ray.TMin)
        {
            float rayLen = length(ray.Direction * payload.colorAndDistance.a);
            float depth = getDepthValue(rayLen);
            // write to depth buffer
            gDepth[launchIndex] = float4(depth, depth, depth, 1.0);
        }
        else if (payload.colorAndDistance.a < 0)
        {
            gDepth[launchIndex] = float4(0, 0, 0, 1);
        }
    }
}
