// shamelessly rewritten from TDA362

#include "Randomness.hlsl"

#define M_PI 3.1415926535

// ----------------
// Material helpers
// ----------------
struct WiSample
{
    float3 wi;
    float3 f;
    float pdf;
};

float3 perpendicular(float3 v)
{
    if (abs(v.x) < abs(v.y))
    {
        return float3(0, -v.z, v.y);
    }
    
    return float3(-v.z, 0, v.x);
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

float2 concentricSampleDisk(uint randState)
{
    float r, theta;
    float sx = 2.0 * random1inclusive(randState) - 1.0;
    float sy = 2.0 * random1inclusive(randState) - 1.0;
    
    if (sx == 0.0 && sy == 0.0)
    {
        return float2(0, 0);
    }
    
    if (sx >= -sy)
    {
        if (sx > sy)
        {
            r = sx;
            if (sy > 0.0)
            {
                theta = sy / r;
            }
            else
            {
                theta = 8.0 + sy / r;
            }
        }
        else
        {
            r = sy;
            theta = 2.0 - sx / r;
        }
    }
    else
    {
        if (sx <= sy)
        {
            r = -sx;
            theta = 4.0 - sy / r;
        }
        else
        {
            r = -sy;
            theta = 6.0 + sx / r;
        }
    }
    
    theta *= M_PI / 4.0;
    return r * float2(cos(theta), sin(theta));
}

float3 cosineSampleHemisphere(uint randState)
{
    float3 ret = float3(concentricSampleDisk(randState), 0.0);
    ret.z = sqrt(max(0.0, 1.0 - ret.x * ret.x - ret.y * ret.y));
    return ret;
}

WiSample sampleHemisphereCosine(float3 wo, float3 n, uint randState)
{
    float3x3 tbn = tangentSpace(n);
    float3 sample = cosineSampleHemisphere(randState);

    WiSample r;
    r.wi = mul(tbn, sample);
    
    if (dot(r.wi, n) > 0.0f)
    {
        r.pdf = max(0.0f, dot(r.wi, n)) / M_PI;
    }
    
    return r;
}

bool sameHemisphere(float3 i, float3 o, float3 n)
{
    return sign(dot(o, n)) == sign(dot(i, n));
}

float fresnel(float3 wi, float3 wo, float fresnelValue)
{
    float3 wh = normalize(wi + wo);
    float F = fresnelValue;
    return float(F + (1.0 - F) * pow(1.0 - dot(wh, wi), 5));
}

// ----------------
// Material classes
// ----------------
class cGlassBTDF
{
    float ior;
    
    float3 f(float3 wi, float3 wo, float3 n)
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
    
    WiSample sampleWi(float3 wo, float3 n, uint randState)
    {
        WiSample r;
        
        float eta = 0.0;
        float3 N;
        if (dot(wo, n) > 0.0)
        {
            N = n;
            eta = 1.0 / ior;
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
};

class cDiffuse
{
    float3 color;
    
    float3 f(float3 wi, float3 wo, float3 n)
    {
        if (dot(wi, n) <= 0.0)
        {
            return float3(0, 0, 0);
        }
        if (!sameHemisphere(wi, wo, n))
        {
            return float3(0, 0, 0);
        }
    
        return (1.0 / M_PI) * color;
    }
    
    WiSample sampleWi(float3 wo, float3 n, uint randState)
    {
        WiSample r = sampleHemisphereCosine(wo, n, randState);
        r.f = f(r.wi, wo, n);
        return r;
    }
};

class cMicrofacetBRDF
{
    float shininess;
    
    float3 f(float3 wi, float3 wo, float3 n)
    {
        float3 wh = normalize(wi + wo);
    
        float s = shininess;
        float d = (s + 2) / (2 * M_PI) * pow(dot(n, wh), s);
        float nwo = dot(n, wh) * dot(n, wo) / dot(wo, wh);
        float nwi = dot(n, wh) * dot(n, wi) / dot(wo, wh);
        float g = min(1.0f, min(nwo, nwi) * 2.0f);
    
        float x = (d * g) / (4 * dot(n, wo) * dot(n, wi));
        return float3(x, x, x);
    }
    
    WiSample sampleWi(float3 wo, float3 n, uint randState)
    {
        float2 rand = float2(0, 0);
        rand.x = random1inclusive(randState);
        rand.y = random1inclusive(randState);
        
        WiSample r;
        float s = shininess;
        float3 tangent = normalize(perpendicular(n));
        float3 bitangent = normalize(cross(tangent, n));
        float phi = 2.0f * M_PI * rand.x;
        float cos_theta = pow(rand.y, 1.0f / (s + 1));
        float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
        float3 wh = normalize(sin_theta * cos(phi) * tangent +
            sin_theta * sin(phi) * bitangent + cos_theta * n);
        float3 wi = -reflect(wo, wh);

        float pdf_wh = (s + 1.0f) * pow(dot(n, wh), s) * (1.0f / (2.0f * M_PI));
        float pdf_wi = pdf_wh / (4.0f * dot(wo, wh));
        r.pdf = pdf_wi;
        r.wi = wi;
        r.f = f(r.wi, wo, n);
        return r;
    }
};

class cLinearBTDF
{
    float transparency;
    cGlassBTDF btdf0;
    cDiffuse btdf1;
    
    float3 f(float3 wi, float3 wo, float3 n)
    {
        float w = transparency;
        return w * btdf0.f(wi, wo, n) + (1.0 - w) * btdf1.f(wi, wo, n);
    }
    
    WiSample sampleWi(float3 wo, float3 n, uint randState)
    {
        float w = transparency;
        float rand = random1inclusive(randState);
        if (rand < w)
        {
            return btdf0.sampleWi(wo, n, randState);
        }
        else
        {
            return btdf1.sampleWi(wo, n, randState);
        }
    }
};

class cDielectricBSDF
{
    float R0;
    cMicrofacetBRDF reflectiveMat;
    cLinearBTDF transmissiveMat;
    
    float3 f(float3 wi, float3 wo, float3 n)
    {
        float3 BRDF = reflectiveMat.f(wi, wo, n);
        float3 BTDF = transmissiveMat.f(wi, wo, n);
        float F = fresnel(wi, wo, R0);

        return F * BRDF + (1.0f - F) * BTDF;
    }
    
    WiSample sampleWi(float3 wo, float3 n, uint randState)
    {
        WiSample r;
        
        float fo = 0;

        if (dot(n, wo) >= 0.0f)
        {
            fo = (R0 + (1.0f - R0) * pow(1.0f - abs(dot(n, wo)), 5));
        }

        float rand = random1inclusive(randState);
        if (rand < fo)
        {
		// Sample the BRDF
            r = reflectiveMat.sampleWi(wo, n, randState);
            r.pdf *= fo;
            float3 wh = normalize(r.wi + wo);
            r.f *= (R0 + (1.0f - R0) * pow(1.0f - abs(dot(wo, wh)), 5));
        }
        else
        {
		// Sample the BTDF
            r = transmissiveMat.sampleWi(wo, n, randState);
            r.pdf *= 1.0f - fo;
            r.f *= 1.0f - fo;
        }

        return r;
    }
};

class cMetalBSDF
{
    float fresnelValue;
    float3 color;
    cMicrofacetBRDF reflectiveMat;
    
    float3 f(float3 wi, float3 wo, float3 n)
    {
        float3 BRDF = reflectiveMat.f(wi, wo, n);
        float F = fresnel(wi, wo, fresnelValue);

        return F * BRDF * color;
    }
    
    WiSample sampleWi(float3 wo, float3 n, uint randState)
    {
        WiSample r = reflectiveMat.sampleWi(wo, n, randState);
        r.f *= fresnel(r.wi, wo, fresnelValue) * color;
        return r;
    }
};

class cLinearBSDF
{
    float metalness;
    cMetalBSDF bsdf0;
    cDielectricBSDF bsdf1;
    
    float3 f(float3 wi, float3 wo, float3 n)
    {
        float3 lhs = bsdf0.f(wi, wo, n);
        float3 rhs = bsdf1.f(wi, wo, n);
        
        float w = metalness;
        return lhs * w + rhs * (1.0 - w);
    }
    
    WiSample sampleWi(float3 wo, float3 n, uint randState)
    {
        float w = metalness;
        float rand = random1inclusive(randState);
        if (rand < w)
        {
            return bsdf0.sampleWi(wo, n, randState);
        }
        else
        {
            return bsdf1.sampleWi(wo, n, randState);
        }
    }
};
