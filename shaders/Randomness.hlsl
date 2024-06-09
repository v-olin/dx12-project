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