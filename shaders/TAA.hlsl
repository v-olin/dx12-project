RWTexture2D<float4> gOutput : register(u0);
RWTexture2D<float4> gHistory : register(u1);
RWTexture2D<float4> gCurrent : register(u2);

[numthreads(32, 32, 1)]
void main(uint3 groupID : SV_GroupID, uint3 tid : SV_DispatchThreadID, uint3 localTID : SV_GroupThreadID, uint groupIndex : SV_GroupIndex)
{
    uint2 threadIdx = uint2(0, 0);
    threadIdx.x = groupID.x * 32 + localTID.x;
    threadIdx.y = groupID.y * 32 + localTID.y;
    
    uint TEX_WIDTH = 0;
    uint TEX_HEIGHT = 0;
    gOutput.GetDimensions(TEX_WIDTH, TEX_HEIGHT);
    
    const float w = 1.0 / 10.0;
    
    if (threadIdx.x < TEX_WIDTH && threadIdx.y < TEX_HEIGHT)
    {
        float3 final = gCurrent[threadIdx].rgb * w + gHistory[threadIdx].rgb * (1.0 - w);
        gOutput[threadIdx] = float4(final, 1);
        gHistory[threadIdx] = float4(final, 1);
    }
}