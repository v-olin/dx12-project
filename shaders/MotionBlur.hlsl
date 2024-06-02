RWTexture2D<float4> inputTexture : register(u0);
RWTexture2D<float4> outputTexture : register(u1);
RWTexture2D<float4> depthBuffer : register(u2);

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
    bool useMotionBlur;
}

[numthreads(32, 32, 1)]
void main(uint3 groupID : SV_GroupID, uint3 tid : SV_DispatchThreadID, uint3 localTID : SV_GroupThreadID, uint groupIndex : SV_GroupIndex)
{
    uint2 threadIdx = uint2(groupID.x * 32 + localTID.x, groupID.y * 32 + localTID.y);
    
    uint TEX_WIDTH, TEX_HEIGHT;
    inputTexture.GetDimensions(TEX_WIDTH, TEX_HEIGHT);
    
    if (threadIdx.x < TEX_WIDTH && threadIdx.y < TEX_HEIGHT)
    {
        // Get depth value of pixel
        float zOverW = depthBuffer[threadIdx].r;
        
        // Convert pixel position to normalized device coordinates (NDC)
        float x = ((threadIdx.x / float(TEX_WIDTH)) * 2.0 - 1.0);
        float y = (1.0 - threadIdx.y / float(TEX_HEIGHT)) * 2.0 - 1.0;
        float4 H = float4(x, y, zOverW, 1.0);
        
        // Transform by inverse view projection matrix to get world position
        float4x4 currViewProjInv = mul(currProjInv, currViewInv);
        float4 worldPos = mul(H, currViewProjInv);
        worldPos /= worldPos.w;
        
        // Transform world position to previous clip space
        float4 prevClipPos = mul(float4(worldPos.xyz, 1.0), mul(prevView, prevProj));
        prevClipPos /= prevClipPos.w;

        // Calculate the velocity vector in screen space
        float2 velocity = (H.xy - prevClipPos.xy) / 2.0f;

        // If velocity is very small, skip motion blur
        if (length(velocity) < 0.001)
        {
            outputTexture[threadIdx] = inputTexture[threadIdx];
            return;
        }

        // Normalize the velocity to make sure we take consistent samples along the path
        velocity = normalize(velocity);

        // Get the current pixel color
        float4 color = inputTexture[threadIdx];
        
        // Accumulate colors along the velocity vector
        float4 accumulatedColor = color;
        int sampleCount = 1;

        // Increase the number of samples and radius for a more noticeable blur
        int numSamples = 10;
        float samplingRadius = 10.0;

        for (int i = 1; i <= numSamples; ++i)
        {
            float2 texCoord = float2(threadIdx) + velocity * (i / float(numSamples)) * samplingRadius;
            if (texCoord.x >= 0 && texCoord.x < TEX_WIDTH && texCoord.y >= 0 && texCoord.y < TEX_HEIGHT)
            {
                accumulatedColor += inputTexture[uint2(texCoord)];
                sampleCount++;
            }
        }
        
        // Average the accumulated color
        float4 finalColor = accumulatedColor / float(sampleCount);
        
        outputTexture[threadIdx] = finalColor;
    }
}