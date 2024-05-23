RWTexture2D<float4> gOutput : register(u0);
RWTexture2D<float4> gInput1 : register(u1);
RWTexture2D<float4> gInput2 : register(u2);
RWTexture2D<float4> gInput3 : register(u3);
RWTexture2D<float4> gInput4 : register(u4);
RWTexture2D<float4> gInput5 : register(u5);
RWTexture2D<float4> gInput6 : register(u6);
RWTexture2D<float4> gInput7 : register(u7);
RWTexture2D<float4> gInput8 : register(u8);
RWTexture2D<float4> gInput9 : register(u9);
RWTexture2D<float4> gInput10 : register(u10);
RWTexture2D<float4> gInput11 : register(u11);
RWTexture2D<float4> gInput12 : register(u12);
RWTexture2D<float4> gInput13 : register(u13);
RWTexture2D<float4> gInput14 : register(u14);
RWTexture2D<float4> gInput15 : register(u15);
RWTexture2D<float4> gInput16 : register(u16);


float3 sampleAt(uint2 coords)
{
    const float w = 1.f / 16.f;
    float4 finalColor = gInput1[coords] * w +
                        gInput2[coords] * w +
                        gInput3[coords] * w +
                        gInput4[coords] * w +
                        gInput5[coords] * w +
                        gInput6[coords] * w +
                        gInput7[coords] * w +
                        gInput8[coords] * w +
                        gInput9[coords] * w +
                        gInput10[coords] * w +
                        gInput11[coords] * w +
                        gInput12[coords] * w +
                        gInput13[coords] * w +
                        gInput14[coords] * w +
                        gInput15[coords] * w +
                        gInput16[coords] * w;
    
    return finalColor.rgb;

}

float3 sampleAt(uint x, uint y)
{
    return sampleAt(uint2(x, y));
}

bool coordsInRange(uint2 coords)
{
    uint TEX_WIDTH = 0;
    uint TEX_HEIGHT = 0;
    gOutput.GetDimensions(TEX_WIDTH, TEX_HEIGHT);
    
    return coords.x < TEX_WIDTH && coords.y < TEX_HEIGHT;
}

[numthreads(32, 32, 1)]
void main(uint3 groupID : SV_GroupID, uint3 tid : SV_DispatchThreadID, uint3 localTID : SV_GroupThreadID, uint groupIndex : SV_GroupIndex)
{
    uint2 threadIdx = uint2(0, 0);
    
    
    threadIdx.x = groupID.x * 32 + localTID.x;
    threadIdx.y = groupID.y * 32 + localTID.y;
    
    const float w = 0.09375;
    float3 final = sampleAt(threadIdx) * 0.25;
    final += sampleAt(threadIdx.x + 1, threadIdx.y) * w;
    final += sampleAt(threadIdx.x - 1, threadIdx.y) * w;
    final += sampleAt(threadIdx.x, threadIdx.y + 1) * w;
    final += sampleAt(threadIdx.x, threadIdx.y - 1) * w;
    
    final += sampleAt(threadIdx.x + 1, threadIdx.y + 1) * w;
    final += sampleAt(threadIdx.x + 1, threadIdx.y - 1) * w;
    final += sampleAt(threadIdx.x - 1, threadIdx.y + 1) * w;
    final += sampleAt(threadIdx.x - 1, threadIdx.y - 1) * w;
    
    gOutput[threadIdx] = float4(final, 1);
}