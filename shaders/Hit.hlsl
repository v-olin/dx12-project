#include "Common.hlsl"

StructuredBuffer<Vertex> Vertices : register(t0);

[shader("closesthit")] void ClosestHit(inout HitInfo payload, Attributes attrib) 
{
    float3 barycentrics = float3(1.f - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);
    uint vertId = 3 * PrimitiveIndex();
    float3 hitColor = Vertices[vertId + 0].color * barycentrics.x +
                  Vertices[vertId + 1].color * barycentrics.y +
                  Vertices[vertId + 2].color * barycentrics.z;
  
    payload.colorAndDistance = float4(hitColor.x, hitColor.y, hitColor.z, RayTCurrent());
}

[shader("closesthit")] void PlaneClosestHit(inout HitInfo payload, Attributes attrib)
{
    payload.colorAndDistance = float4(0, 0, 0, RayTCurrent());
}
