RWTexture2D<float4> gOutput : register(u0);
RWTexture2D<float4> gHistory : register(u1);
RWTexture2D<float4> gCurrent : register(u2);
RWTexture2D<float4> gDepth : register(u3);

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

bool posIsValid(uint2 pos)
{
    uint xdim, ydim;
    gDepth.GetDimensions(xdim, ydim);
    
    if (pos.x >= 0 && pos.x < xdim && pos.y >= 0 && pos.y < ydim)
    {
        return true;
    }
    
    return false;
}

float getRayLen(uint2 currPos)
{
    float linearDepth = gDepth[currPos].r * farPlane;
    // this is sus
    float depth = (((farPlane + nearPlane - ((2 * nearPlane * farPlane) / linearDepth)) / (farPlane * nearPlane)) + 1.0) / 2.0;
    float rayLen = 1 / (((1 / farPlane) - (1 / nearPlane)) * depth + (1 / nearPlane));
    return rayLen;
}

float3 normalizeZ(float3 vec)
{
    float scale = 1.0 / vec.z;
    return float3(vec.x * scale, vec.y * scale, 1.0);
}

uint2 getPrevFramePos(uint2 launchIndex)
{
    uint xdim, ydim;
    gDepth.GetDimensions(xdim, ydim);
    float2 dims = float2(xdim, ydim);
    
    // this is the target pixel this thread's ray was shot through in current camera space
    float2 d = (((launchIndex.xy + 0.5f) / dims.xy) * 2.f - 1.f);
    float4 target = mul(currProjInv, float4(d.x, -d.y, 1, 1));
    
    // this is the ray description of this thread's pixel's ray
    float3 rayDir = mul(currViewInv, float4(target.xyz, 0)).xyz;
    float rayLen = getRayLen(launchIndex);
    float3 rayOrigin = mul(currViewInv, float4(0, 0, 0, 1)).xyz;
    
    // this is where the thread's ray ends up
    float3 rayEndPos = rayOrigin + rayDir * rayLen;
    
    float4 camRayEnd = mul(prevProj, mul(prevView, float4(rayEndPos, 1.0)));
    
    if (camRayEnd.w == 0.0)
    {
        // invalid pos
        return uint2(~0, ~0);
    }
    
    float3 ndc = camRayEnd.xyz / camRayEnd.w;
    ndc.y = -ndc.y;
    float2 prevFrameScreenPos = ((ndc.xy + 1.0) / 2.0) * dims; // + (dims / 3.0);
    
    /*
    float3 prevCamPos = mul(prevViewInv, float4(0, 0, 0, 1)).xyz;
    float3 prevFrameRayDir = normalizeZ(rayEndPos - prevCamPos);
    float4 prevFrameRayTarget = mul(prevView, float4(prevFrameRayDir, 1.0));
    float4 prevFrameScreenPos = mul(prevProj, prevFrameRayTarget);
    */
    
    uint2 prevPos = float2(prevFrameScreenPos.xy);
    
    if (posIsValid(prevPos))
    {
        return prevPos;
    }
    
    return uint2(~0, ~0);
}

[numthreads(32, 32, 1)]
void main(uint3 groupID : SV_GroupID, uint3 tid : SV_DispatchThreadID, uint3 localTID : SV_GroupThreadID, uint groupIndex : SV_GroupIndex)
{
    uint2 threadIdx = uint2(0, 0);
    threadIdx.x = groupID.x * 32 + localTID.x;
    threadIdx.y = groupID.y * 32 + localTID.y;
    
    uint TEX_WIDTH = 0;
    uint TEX_HEIGHT = 0;
    gOutput.GetDimensions(TEX_WIDTH, TEX_HEIGHT);
    
    const float w = 1.0 / 8.0;
    
    if (threadIdx.x < TEX_WIDTH && threadIdx.y < TEX_HEIGHT)
    {
        float4 blended = float4(0, 0, 0, 1);
        uint2 prevFramePos = getPrevFramePos(threadIdx);
        if (posIsValid(prevFramePos))
        {
            float4 currCol = gCurrent[threadIdx];
            float4 histCol = gHistory[prevFramePos];
            
            if (length(gDepth[threadIdx].rgb) == 0.0)
            {
                blended = float4(gCurrent[threadIdx].rgb, 1.0);
            }
            else
            {
                blended = gCurrent[threadIdx] * w + gHistory[prevFramePos] * (1.0 - w);
            }
        }
        else
        {
            blended = gCurrent[threadIdx];
        }
        
        float4 final = float4(blended.rgb, 1.0);
        
        gOutput[threadIdx] = final;
    }
}