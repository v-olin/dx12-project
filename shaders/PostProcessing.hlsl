// Define the texture buffer
RWTexture2D<float4> textureBuffer1 : register(u0);
RWTexture2D<float4> textureBuffer2 : register(u1);
RWTexture2D<float4> textureBuffer3 : register(u2);

void Threshold(uint2 threadIdx)
{
    // Read the pixel color from the texture
    float4 pixelColor = textureBuffer1[threadIdx];
    
    float brightness = dot(pixelColor.rgb, float3(0.299, 0.587, 0.114));
    if (brightness < 0.5)
    {
        pixelColor.rgb = 0.0;
    }
    
    textureBuffer2[threadIdx] = pixelColor;
}


void Blur(uint2 threadIdx, uint texWidth, uint texHeight)
{
    float4 color = float4(0, 0, 0, 0);
    int radius = 4; // Example blur radius, you can increase this

    // Compute the normalization factor for the kernel
    float kernelSum = 0.0;
    float kernelSize = (2 * radius + 1) * (2 * radius + 1);
    float weight = 1.0 / kernelSize;

    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            int2 offset = int2(x, y);
            uint2 samplePos = threadIdx + offset;
            // Clamp samplePos to texture dimensions
            samplePos = clamp(samplePos, uint2(0, 0), uint2(texWidth, texHeight) - uint2(1, 1));
            color += textureBuffer2[samplePos] * weight;
            kernelSum += weight;
        }
    }

    // Normalize the color by the sum of the kernel weights
    color /= kernelSum;
    textureBuffer3[threadIdx] = color;
    
    color = float4(0, 0, 0, 0);
    kernelSum = 0.0;
    
    for (int z = -radius; z <= radius; ++z)
    {
        for (int i = -radius; i <= radius; ++i)
        {
            int2 offset = int2(i, z);
            uint2 samplePos = threadIdx + offset;
            // Clamp samplePos to texture dimensions
            samplePos = clamp(samplePos, uint2(0, 0), uint2(texWidth, texHeight) - uint2(1, 1));
            color += textureBuffer3[samplePos] * weight;
            kernelSum += weight;
        }
    }
    
    color /= kernelSum;
    textureBuffer2[threadIdx] = color;
}


[numthreads(32, 32, 1)]
void main(uint3 groupID : SV_GroupID, uint3 tid : SV_DispatchThreadID, uint3 localTID : SV_GroupThreadID, uint groupIndex : SV_GroupIndex)
{
    // Get the size of the texture
    uint texWidth;
    uint texHeight;
    textureBuffer1.GetDimensions(texWidth, texHeight);
    
    // Init a thread index
    uint2 threadIdx = uint2(0, 0);

    uint threadColumn = groupID.x * 32 + localTID.x;
    uint threadRow = groupID.y * 32 + localTID.y;

    // Make sure the current thread is within the bounds of the texture
    if (threadColumn < texWidth && threadRow < texHeight)
    {
        // If the thread is within the bounds of the texture, set the thread index
        threadIdx.x = threadColumn;
        threadIdx.y = threadRow;
        
        Threshold(threadIdx);
        
        Blur(threadIdx, texWidth, texHeight);
        
        Blur(threadIdx, texWidth, texHeight);
        
        float4 baseColor = textureBuffer1[threadIdx];
        float4 bloomColor = textureBuffer2[threadIdx];
        textureBuffer1[threadIdx] = baseColor + bloomColor * 0.5f;
    }
}