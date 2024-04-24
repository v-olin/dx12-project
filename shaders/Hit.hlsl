#include "Common.hlsl"

[shader("closesthit")] void ClosestHit(inout HitInfo payload, Attributes attrib) 
{
  payload.colorAndDistance = float4(1, 1, 0, RayTCurrent());
}

[shader("closesthit")] void PlaneClosestHit(inout HitInfo payload, Attributes attrib)
{
    payload.colorAndDistance = float4(1, 0, 1, RayTCurrent());
}
