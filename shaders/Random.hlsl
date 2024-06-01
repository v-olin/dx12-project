#include "Randomness.hlsl"

RWTexture2D<float4> gOutput : register(u0);

cbuffer NoiseConstBuff : register(b0)
{
    uint frameNr;
}

[numthreads(32, 32, 1)]
void main(uint3 groupID : SV_GroupID, uint3 tid : SV_DispatchThreadID, uint3 localTID : SV_GroupThreadID, uint groupIndex : SV_GroupIndex)
{
    uint2 threadIdx = uint2(0, 0);
    
    uint TEX_WIDTH = 0;
    uint TEX_HEIGHT = 0;
    gOutput.GetDimensions(TEX_WIDTH, TEX_HEIGHT);
    
    threadIdx.x = groupID.x * 32 + localTID.x;
    threadIdx.y = groupID.y * 32 + localTID.y;
    
    if (threadIdx.x < TEX_WIDTH && threadIdx.y < TEX_HEIGHT)
    {
        uint threadSeed = (TEX_WIDTH * TEX_HEIGHT * frameNr) + threadIdx.x * TEX_WIDTH + threadIdx.y;
        uint threadState = seedThread(threadSeed);
        float r = random1inclusive(threadState);
        float g = random1inclusive(threadState);
        float b = random1inclusive(threadState);
        
        gOutput[threadIdx] = float4(r, g, b, 1);
    }
}