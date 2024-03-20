#pragma once

#include "PathWin.h"

#include <d3d12.h>

#include <exception>

#pragma comment(lib, "d3d12.lib")

#define THROW_IF_FAILED(hrcall) if(FAILED(hr = (hrcall))) { throw std::exception(); }


#include "../../vendor/SimpleMath/SimpleMath.h"

using float2 = DirectX::SimpleMath::Vector2;
using float3 = DirectX::SimpleMath::Vector3;
using float4 = DirectX::SimpleMath::Vector4;
using float4x4 = DirectX::SimpleMath::Matrix;

namespace pathtracex {

	inline D3D12_RESOURCE_BARRIER transitionBarrierFromRenderTarget(ID3D12Resource* pResource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) noexcept {
		D3D12_RESOURCE_BARRIER rbarr{};
		ZeroMemory(&rbarr, sizeof(rbarr));
		rbarr.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		rbarr.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		rbarr.Transition.pResource = pResource;
		rbarr.Transition.StateBefore = before;
		rbarr.Transition.StateAfter = after;
		rbarr.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		return rbarr;
	}

	inline D3D12_HEAP_PROPERTIES getDefaultHeapProperties() noexcept {
		D3D12_HEAP_PROPERTIES hprops{};
		hprops.Type = D3D12_HEAP_TYPE_UPLOAD;
		hprops.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
		hprops.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
		hprops.CreationNodeMask = 1u;
		hprops.VisibleNodeMask = 1u;
		return hprops;
	}

	inline D3D12_RESOURCE_DESC getResourceDescriptionFromSize(UINT64 bufferSize) noexcept {
		D3D12_RESOURCE_DESC rdesc{};
		rdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		rdesc.Alignment = 0ui64;
		rdesc.Width = bufferSize;
		rdesc.Height = 1u;
		rdesc.DepthOrArraySize = 1u;
		rdesc.MipLevels = 1u;
		rdesc.Format = DXGI_FORMAT_UNKNOWN;
		rdesc.SampleDesc.Count = 1u;
		rdesc.SampleDesc.Quality = 0u;
		rdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		rdesc.Flags = D3D12_RESOURCE_FLAG_NONE;
		return rdesc;
	}

	inline D3D12_RASTERIZER_DESC getRasterizerDesc() noexcept {
		return {
			D3D12_FILL_MODE_SOLID,
			D3D12_CULL_MODE_BACK,
			FALSE,
			D3D12_DEFAULT_DEPTH_BIAS,
			D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
			D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
			TRUE, FALSE, FALSE,
			0, D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
		};
	}

	inline D3D12_BLEND_DESC getDefaultBlendDesc() noexcept {
		D3D12_RENDER_TARGET_BLEND_DESC drtbd{
			FALSE, FALSE,
			D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
			D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
			D3D12_LOGIC_OP_NOOP,
			D3D12_COLOR_WRITE_ENABLE_ALL
		};

		D3D12_BLEND_DESC blendd{ FALSE, FALSE, drtbd };
		for (UINT i = 0; i < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) {
			blendd.RenderTarget[i] = drtbd;
		}

		return blendd;
	}

	inline D3D12_ROOT_SIGNATURE_DESC getRootSignatureDesc() noexcept {
		D3D12_ROOT_SIGNATURE_DESC rootSignDesc{};
		rootSignDesc.NumParameters = 0;
		rootSignDesc.pParameters = nullptr;
		rootSignDesc.NumStaticSamplers = 0;
		rootSignDesc.pStaticSamplers = nullptr;
		rootSignDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
		return rootSignDesc;
	}

	inline D3D12_GRAPHICS_PIPELINE_STATE_DESC getPreparedPipeStateDesc() noexcept {
		D3D12_GRAPHICS_PIPELINE_STATE_DESC psd{};
		psd.RasterizerState = getRasterizerDesc();
		psd.BlendState = getDefaultBlendDesc();
		psd.DepthStencilState.DepthEnable = FALSE;
		psd.DepthStencilState.StencilEnable = FALSE;
		psd.SampleMask = UINT_MAX;
		psd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		psd.NumRenderTargets = 1;
		psd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
		psd.SampleDesc.Count = 1;
		return psd;
	}
}
