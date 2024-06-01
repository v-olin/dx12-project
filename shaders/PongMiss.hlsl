#include "Common.hlsl"

[shader("miss")]
void Miss(inout PathHitInfo payload : SV_RayPayload)
{
    float3 unitDir = normalize(WorldRayDirection());
    float w = 0.5 * (unitDir.y + 1.0);
    
    float3 skyCol = (1.0 - w) * float3(1, 1, 1) + w * float3(0.5, 0.7, 1.0);
    payload.colorAndDistance = float4(skyCol, -1.0);
}