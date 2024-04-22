struct VS_INPUT
{
    float4 pos : POSITION;
    float4 color : COLOR;
    float2 texCoord : TEXCOORD;
    float3 normal : NORMAL;
};

struct PointLight
{
    float4 position;
};

struct VS_OUTPUT
{
    float4 pos: SV_POSITION;
    float4 color : COLOR;
    float2 texCoord : TEXCOORD;
    float4 worldNormal : WORLDNORMAL;
    float4 worldPos : WORLDPOS;
};

cbuffer ConstantBuffer : register(b0)
{
    float4x4 wvpMat; // 64 bytes
    float4x4 modelMatrix; // 64 bytes
    float4x4 normalMatrix; // 64 bytes
    PointLight pointLights[3]; // 48 bytes
    int pointLightCount; // 4 bytes
    bool hasColTex; // 1 bytes
    bool hasNormalTex;
    bool hasShinyTex;
    float material_shininess;
    float material_metalness;
    float material_fresnel;
  };


VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;

    output.pos = mul(input.pos, wvpMat);
    output.color = input.color;
    output.texCoord = input.texCoord;

    float4 normal = float4(input.normal, 0.0f);
    output.worldNormal = normalize(mul(normal, normalMatrix));
    output.worldPos = mul(input.pos, modelMatrix);
        return output;
}