#include "Common.hlsl"
//#include "Sampling.hlsl"

#define MAX_BOUNCES 5
#define EPSILON 0.01
#define RAY_MAX 10000.0
#define M_PI 3.1415926535

struct ShadowHitInfo
{
    bool isHit;
};

struct WiSample
{
    float3 wi;
    float3 f;
    float pdf;
};

struct PointLight
{
    float4 position;
    float4 color;
    float intensity;
};

struct MeshData
{
    float4 material_color;
    float4 material_emmision;
    float4x4 normal_matrix;
    bool hasColTex;
    bool hasNormalTex;
    bool hasShinyTex;
    bool hasMetalTex;
    bool hasFresnelTex;
    bool hasEmisionTex;
    float material_shininess;
    float material_metalness;
    float material_fresnel;
    float material_transparency;
    float material_ior;
    bool hasMaterial;
};

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

RWTexture2D<float4> noiseTex : register(u0);
StructuredBuffer<Vertex> Vertices : register(t0);
StructuredBuffer<int> indices : register(t1);
RaytracingAccelerationStructure SceneBVH : register(t2);
StructuredBuffer<MeshData> meshdatas : register(t3);

cbuffer LightBuffer : register(b1)
{
    //PointLight lights[3];
    float4 position0;
    float4 position1;
    float4 position2;
    float4 colorIntense0;
    float4 colorIntense1;
    float4 colorIntense2;
    //int lightCount;
}

float3 HitAttribute(float3 vertexAttribute[3], Attributes attr)
{
    return vertexAttribute[0] +
        attr.bary.x * (vertexAttribute[1] - vertexAttribute[0]) +
        attr.bary.y * (vertexAttribute[2] - vertexAttribute[0]);
}

float3 HitWorldPosition()
{
    return WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
}

float3 GetPointNormal(Attributes attrib)
{
    uint vertId = 3 * PrimitiveIndex();
    
    float3 normals[3] =
    {
        Vertices[vertId + 0].normal,
        Vertices[vertId + 1].normal,
        Vertices[vertId + 2].normal
    };
    
    return normalize(HitAttribute(normals, attrib));

}

float3 GetMaterialColor(Attributes attrib)
{
    uint vertId = 3 * PrimitiveIndex();
    uint matidx = Vertices[vertId].materialIdx;
    
    if (meshdatas[matidx].hasMaterial)
    {
        float3 colors[3] =
        {
            meshdatas[Vertices[vertId + 0].materialIdx].material_color.rgb,
            meshdatas[Vertices[vertId + 1].materialIdx].material_color.rgb,
            meshdatas[Vertices[vertId + 2].materialIdx].material_color.rgb
        };
        
        float3 hitColor = HitAttribute(colors, attrib);
        return hitColor;
    }
    else
    {
        float3 colors[3] =
        {
            float3(1, 0, 0),
            float3(0, 1, 0),
            float3(0, 0, 1)
        };
        
        // blend them with bary
        float3 hitColor = HitAttribute(colors, attrib);
        return hitColor;
    }
}

float3 GetInitialRandomVector()
{
    return normalize(noiseTex[DispatchRaysIndex().xy].rgb);
}

float3 GetSecondRandomVector()
{
    uint texWidth = 0;
    uint texHeight = 0;
    noiseTex.GetDimensions(texWidth, texHeight);
    float2 rand = GetInitialRandomVector().xy;
    
    uint2 newRand = uint2(rand.x * float(texWidth), rand.y * float(texHeight));
    return noiseTex[newRand].rgb;
}

float3 GetThirdRandomVector()
{
    
    uint texWidth = 0;
    uint texHeight = 0;
    noiseTex.GetDimensions(texWidth, texHeight);
    float2 rand = GetInitialRandomVector().yz;
    
    uint2 newRand = uint2(rand.x * float(texWidth), rand.y * float(texHeight));
    return noiseTex[newRand].rgb;
}

float3 perpendicular(float3 v)
{
    if (abs(v.x) < abs(v.y))
    {
        return float3(0.0f, -v.z, v.y);
    }
    return float3(-v.z, 0.0f, v.x);
}

float3 NewRandomVector(float3 oldVec)
{
    return float3(0, 0, 0);
}

float3 ToSameHemisphere(float3 v, float3 n)
{
    if (dot(v, n) > 0.0)
    {
        return v;
    }
    else
    {
        return -v;
    }
}

float Fresnel(uint matIdx, float3 wi, float3 wo)
{
    float3 wh = normalize(wi + wo);
    float R0 = meshdatas[matIdx].material_fresnel;
    return float(R0 + (1.0 - R0) * pow(1.0 - dot(wh, wi), 5));
}

bool sameHemisphere(float3 wi, float3 wo, float3 n)
{
    return sign(dot(wo, n)) == sign(dot(wi, n));
}

float3x3 tangentSpace(float3 n)
{
    float sign = 0.0;
    if (n.z < 0.0)
    {
        sign = -1.0;
    }
    else if (n.z > 0.0)
    {
        sign = 1.0;
    }

    const float a = -1.0 / (sign + n.z);
    const float b = n.x * n.y * a;
    float3x3 r;
    r[0] = float3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    r[1] = float3(b, sign + n.y * n.y * a, -n.y);
    r[2] = n;
    return r;
}

float2 concentricSampleDisk()
{
    float2 randU = GetInitialRandomVector().xy;
    
    float r, theta;
    float u1 = randU.r;
    float u2 = randU.g;
    float sx = 2 * u1 - 1;
    float sy = 2 * u2 - 1;
    if (sx == 0.0 && sy == 0.0)
    {
        return float2(0, 0);
    }
    if (sx >= -sy)
    {
        if (sx > sy)
        { // Handle first region of disk
            r = sx;
            if (sy > 0.0)
                theta = sy / r;
            else
                theta = 8.0f + sy / r;
        }
        else
        { // Handle second region of disk
            r = sy;
            theta = 2.0f - sx / r;
        }
    }
    else
    {
        if (sx <= sy)
        { // Handle third region of disk
            r = -sx;
            theta = 4.0f - sy / r;
        }
        else
        { // Handle fourth region of disk
            r = -sy;
            theta = 6.0f + sx / r;
        }
    }
    theta *= float(M_PI) / 4.0f;
    return r * float2(cos(theta), sin(theta));
}

float3 cosineSampleHemisphere()
{
    float3 ret = float3(concentricSampleDisk(), 0.0);
    ret.z = sqrt(max(0.0, 1.0 - ret.x * ret.x - ret.y * ret.y));
    return ret;
}

WiSample SampleHemisphereCosine(float3 wo, float3 n)
{
    float3x3 tbn = tangentSpace(n);
    float3 sample = cosineSampleHemisphere();
    WiSample r;
    r.wi = mul(tbn, sample);
    if (dot(r.wi, n) > 0.0f)
    {
        r.pdf = max(0.0f, dot(r.wi, n)) / M_PI;
    }
    return r;
}

float3 SampleDiffuseBTDF(uint matIdx, float3 wi, float3 wo, float3 n)
{
    if (dot(wi, n) <= 0.0)
    {
        return float3(0, 0, 0);
    }
    
    if (!sameHemisphere(wi, wo, n))
    {
        return float3(0, 0, 0);
    }
    
    float3 color = meshdatas[matIdx].material_color;
    return (1.0 / M_PI) * color;
}

float3 SampleGlassBTDF(float3 wi, float3 wo, float3 n)
{
    if (sameHemisphere(wi, wo, n))
    {
        return float3(0, 0, 0);
    }
    else
    {
        return float3(1, 1, 1);
    }

}

float3 SampleLinearBTDF(uint matIdx, float3 wi, float3 wo, float3 n)
{
    float t = meshdatas[matIdx].material_transparency;
    float3 btdf0 = SampleGlassBTDF(wi, wo, n);
    float3 btdf1 = SampleDiffuseBTDF(matIdx, wi, wo, n);
    
    return t * btdf0 + (1.0 - t) * btdf1;
}

float3 SampleMicrofacetBRDF(uint matIdx, float3 wi, float3 wo, float3 n)
{
    float3 wh = normalize(wi + wo);
    
    float s = meshdatas[matIdx].material_shininess;
    float d = (s + 2.0) / (2.0 * M_PI) * pow(dot(n, wh), s);
    float nwo = dot(n, wh) * dot(n, wo) / dot(wo, wh);
    float nwi = dot(n, wh) * dot(n, wi) / dot(wo, wh);
    float g = min(1.0, min(nwo, nwi) * 2.0);
    
    float x = (d * g) / (4.0 * dot(n, wo) * dot(n, wi));
    
    return float3(x, x, x);
}

float3 SampleMetalBSDF(uint matIdx, float3 wi, float3 wo, float3 n)
{
    float3 brdf = SampleMicrofacetBRDF(matIdx, wi, wo, n);
    float f = Fresnel(matIdx, wi, wo);
    float3 color = meshdatas[matIdx].material_color.rgb;
    
    return f * brdf * color;
}

float3 SampleDielectricBSDF(uint matIdx, float3 wi, float3 wo, float3 n)
{
    float3 brdf = SampleMicrofacetBRDF(matIdx, wi, wo, n);
    float3 btdf = SampleLinearBTDF(matIdx, wi, wo, n);
    float F = Fresnel(matIdx, wi, wo);
    
    return F * brdf + (1.0 - F) * btdf;
}

float3 SampleMaterial(Attributes attrib, float3 wi, float3 wo, float3 n)
{
    uint vertId = 3 * PrimitiveIndex();
    uint matIdx = Vertices[vertId].materialIdx;
    MeshData mat = meshdatas[matIdx];
    
    float m = mat.material_metalness;
    
    float3 lhs = SampleMetalBSDF(matIdx, wi, wo, n);
    float3 rhs = SampleDielectricBSDF(matIdx, wi, wo, n);
 
    return lhs * m + rhs * (1.0 - m);
}

WiSample WiSampleMicrofacetBRDF(uint matIdx, float3 wo, float3 n)
{
    float2 randV = GetSecondRandomVector().xy;
    
    WiSample r;
    float s = meshdatas[matIdx].material_shininess;
    float3 tangent = normalize(perpendicular(n));
    float3 bitangent = normalize(cross(tangent, n));
    float phi = 2.0f * M_PI * randV.x;
    float cos_theta = pow(randV.y, 1.0f / (s + 1));
    float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
    float3 wh = normalize(
		sin_theta * cos(phi) * tangent + sin_theta * sin(phi) * bitangent
		+ cos_theta * n
	);
    float3 wi = -reflect(wo, wh);

    float pdf_wh = (s + 1.0f) * pow(dot(n, wh), s) * (1.0f / (2.0f * M_PI));
    float pdf_wi = pdf_wh / (4.0f * dot(wo, wh));
    r.pdf = pdf_wi;
    r.wi = wi;
    r.f = SampleMicrofacetBRDF(matIdx, r.wi, wo, n);
    return r;
}

WiSample WiSampleGlassBTDF(uint matIdx, float3 wo, float3 n)
{
    WiSample r;

    float ior = meshdatas[matIdx].material_ior;
    float eta;
    float3 N;
    if (dot(wo, n) > 0.0f)
    {
        N = n;
        eta = 1.0f / ior;
    }
    else
    {
        N = -n;
        eta = ior;
    }

    float w = dot(wo, N) * eta;
    float k = 1.0f + (w - eta) * (w + eta);
    if (k < 0.0f)
    {
		// Total internal reflection
        r.wi = reflect(-wo, n);
    }
    else
    {
        k = sqrt(k);
        r.wi = normalize(-eta * wo + (w - k) * N);
    }
    r.pdf = abs(dot(r.wi, n));
    r.f = float3(1.0f, 1.0f, 1.0f);

    return r;
}

WiSample WiSampleDiffuseBTDF(uint matIdx, float3 wo, float3 n)
{
    WiSample r = SampleHemisphereCosine(wo, n);
    r.f = SampleDiffuseBTDF(matIdx, r.wi, wo, n);
    return r;
}

WiSample WiSampleLinearBTDF(uint matIdx, float3 wo, float3 n)
{
    WiSample r;
    
    float rand = GetInitialRandomVector().b;
    float w = meshdatas[matIdx].material_transparency;
    
    if (rand < w)
    {
        return WiSampleGlassBTDF(matIdx, wo, n);
    }
    else
    {
        return WiSampleDiffuseBTDF(matIdx, wo, n);
    }
    
    return r;
}

WiSample WiSampleDielectricBSDF(uint matIdx, float3 wo, float3 n)
{
    WiSample r;
    float fo = 0;
    float R0 = meshdatas[matIdx].material_fresnel;
    float rand = GetInitialRandomVector().g;
    
    if (dot(n, wo) >= 0.0f)
    {
        fo = (R0 + (1.0f - R0) * pow(1.0f - abs(dot(n, wo)), 5));
    }

    if (rand < fo)
    {
		// Sample the BRDF
        r = WiSampleMicrofacetBRDF(matIdx, wo, n);
        r.pdf *= fo;
        float3 wh = normalize(r.wi + wo);
        r.f *= (R0 + (1.0f - R0) * pow(1.0f - abs(dot(wo, wh)), 5));
    }
    else
    {
		// Sample the BTDF
        r = WiSampleLinearBTDF(matIdx, wo, n);
        r.pdf *= 1.0f - fo;
        r.f *= 1.0f - fo;
    }
    
    return r;
}

WiSample WiSampleMetalBSDF(uint matIdx, float3 wo, float3 n)
{
    WiSample r;
    
    r = WiSampleMicrofacetBRDF(matIdx, wo, n);
    float f = Fresnel(matIdx, r.wi, wo);
    r.f *= f * meshdatas[matIdx].material_color;
    
    return r;
}

WiSample WiSampleMaterial(uint matIdx, float3 wo, float3 n)
{
    float randf = GetInitialRandomVector().r;
    float w = meshdatas[matIdx].material_metalness;
    
    if (randf < w)
    {
        return WiSampleMetalBSDF(matIdx, wo, n);
    }
    else
    {
        return WiSampleDielectricBSDF(matIdx, wo, n);
    }
}

float3 GetLightPos(int index)
{
    if (index == 0)
    {
        return position0.xyz;
    }
    else if (index == 1)
    {
        return position1.xyz;
    }
    else
    {
        return position2.xyz;
    }
}

float4 GetLightProps(int index)
{
    if (index == 0)
    {
        return colorIntense0;
    }
    else if (index == 1)
    {
        return colorIntense1;
    }
    else
    {
        return colorIntense2;
    }
}

float3 DirectIllumination(Attributes attrib, float3 hitNormal)
{
    float3 dillum = float3(0, 0, 0);
    float3 hitPos = HitWorldPosition();
    float3 hitWo = normalize(WorldRayDirection() * -1.0);
    
    for (int i = 0; i < 3; ++i)
    {
        float3 lightPos = GetLightPos(i);
        float4 lightProp = GetLightProps(i);
        float3 lightDiff = lightPos - hitPos;
        
        if (length(lightDiff) > 20 && i > 0)
        {
            continue;
        }
        
        float3 wi = normalize(lightDiff);

        RayDesc shadowRay;
        shadowRay.Origin = hitPos + 0.01 * wi;
        shadowRay.Direction = wi;
        shadowRay.TMin = EPSILON;
        shadowRay.TMax = length(lightDiff);

        ShadowHitInfo shadowPayload;
        shadowPayload.isHit = false;
        
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, shadowRay, shadowPayload);
        
        if (shadowPayload.isHit == false)
        {
            float3 lightcol = lightProp.rgb;
            float dlight = length(lightDiff);
            float falloff = 1.0 / (dlight * dlight);

            float3 Li = lightProp.a * falloff * lightcol;
            float3 matF = SampleMaterial(attrib, wi, hitWo, hitNormal);
        
            dillum += matF * Li * max(0.0, dot(wi, hitNormal));
        }
    }
    
    return float3(clamp(dillum.r, 0.0, 1.0), clamp(dillum.g, 0.0, 1.0), clamp(dillum.b, 0.0, 1.0));
    //return float3(min(1.0, dillum.r), min(1.0, dillum.g), min(1.0, dillum.b));
}

[shader("closesthit")]
void ClosestHit(inout PathHitInfo payload, Attributes attrib)
{
    float rayLen = RayTCurrent();
    float3 L = float3(0, 0, 0);
    float3 ptp = float3(1, 1, 1);
    
    uint vertId = 3 * PrimitiveIndex();
    uint matIdx = Vertices[vertId].materialIdx;
    float3 hitN = GetPointNormal(attrib);
    hitN = normalize(mul(float4(normalize(hitN), 0.0f), meshdatas[matIdx].normal_matrix).xyz);
    
    float3 dillum = DirectIllumination(attrib, hitN);
    float3 emission = meshdatas[matIdx].material_emmision;
    
    L += ptp * (dillum); //  + emission
        
    payload.colorAndDistance = float4(L, rayLen);
}

// i think this isn't used?
[shader("closesthit")]
void PlaneClosestHit(inout PathHitInfo payload, Attributes attrib)
{
    payload.colorAndDistance = float4(0, 1, 0, RayTCurrent());
    
    /*
    float shadowFactor = 0.0f;
    
    for (int i = 0; i < lightCount && i < 1; i++)
    {
        PointLight plight = lights[i];
        float3 lightPos = float3(plight.position.x, plight.position.y, plight.position.z);
        
        float3 worldOrigin = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
        float3 lightDir = normalize(lightPos - worldOrigin);
        
        RayDesc ray;
        ray.Origin = worldOrigin;
        ray.Direction = lightDir;
        ray.TMin = 0.01f;
        ray.TMax = 100000;
        bool hit = true;
        
        ShadowHitInfo shadowPayload;
        shadowPayload.isHit = false;
        
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
        
        shadowFactor += shadowPayload.isHit ? 0.7f : 0.0f;
    }
    
    float3 color = float3(1, 1, 1) * (1 - shadowFactor);
    
    payload.colorAndDistance = float4(color.r, color.g, color.b, RayTCurrent());
    */
}
