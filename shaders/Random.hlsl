RWTexture2D<float4> gOutput : register(u0);

// PRNG stuff

uint seedThread(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

// [0, ~0]
uint random(inout uint state)
{
    state ^= (state << 13);
    state ^= (state >> 17);
    state ^= (state << 5);
    return state;
}

// [0.0, 1.0)
float random1(inout uint state)
{
    return asfloat(0x3f800000 | random(state) >> 9) - 1.0;
}

// [0.0, 1.0]
float random1inclusive(inout uint state)
{
    return random(state) / float(0xffffffff);
}

uint random(inout uint state, uint lower, uint upper)
{
    return lower + uint(float(upper - lower + 1) * random1(state));
}

// CS stuff

#define TEX_WIDTH 1280
#define TEX_HEIGHT 720

[numthreads(32, 32, 1)]
void main(uint3 groupID : SV_GroupID, uint3 tid : SV_DispatchThreadID, uint3 localTID : SV_GroupThreadID, uint groupIndex : SV_GroupIndex)
{
    uint2 threadIdx = uint2(0, 0);
    
    uint threadColumn = groupID.x * 32 + localTID.x;
    uint threadRow = groupID.y * 32 + localTID.y;
    
    if (threadColumn < TEX_WIDTH && threadRow < TEX_HEIGHT)
    {
        gOutput[threadIdx] = float4(0.5, 0.7, 0.1, 1);
    }
}