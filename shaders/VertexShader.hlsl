Texture2D normalTex : register(t1);
SamplerState s1 : register(s0);
struct VS_INPUT
{
    float3 pos : POSITION;
    float4 color : COLOR;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD;
    float3 tangent : TANGENT;
    float3 biTangent : BITANGENT;
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
    float4 viewSpaceNormal : VIEWSPACENORMAL;
    float4 viewSpacePos : VIEWSPACEPOS;
    float3 tangent :TANGENT;
};

cbuffer ConstantBuffer : register(b0)
{
    float4x4 wvpMat; // 64 bytes
    float4x4 normalMatrix;
    float4x4 modelViewMatrix;
    float4x4 viewInverse;
    float4x4 viewMat;
    
    PointLight pointLights[3]; // 48 bytes
    int pointLightCount; // 4 bytes
    bool isProcWorld;
};

cbuffer ConstantMeshBuffer : register(b1)
{
	float4 material_emmision;
    float4 material_color;
    bool hasColTex; // 1 bytes
    bool hasNormalTex;
    bool hasShinyTex;
	bool hasMetalTex;
	bool hasFresnelTex;
	bool hasEmisionTex;
    float material_shininess;
    float material_metalness;
    float material_fresnel;
    bool hasMaterial;
		
    //for proc world only, should maybe not be here...
	float stop_flat;
	float stop_interp;
}

VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;

    output.pos = mul(float4(input.pos, 1.0), wvpMat);
    output.color = input.color;
    output.texCoord = input.texCoord;

    //float4 normal = float4(input.normal, 0.0f);
    //output.worldNormal = normalize(mul(normal, normalMatrix));
    //output.worldPos = mul(float4(input.pos, 1.0), modelMatrix);
    
    output.viewSpaceNormal = mul(float4(input.normal, 0.0), normalMatrix);
    output.viewSpacePos = mul(float4(input.pos, 1.0), modelViewMatrix);
    output.tangent = mul(float4(input.tangent, 0.0), normalMatrix).xyz;

    
    
    return output;
}