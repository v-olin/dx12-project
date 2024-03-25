#pragma once
#include <vector>
#include "Vertex.h"
#include <DirectXMath.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <d3d12.h>

namespace pathtracex {
	class DXIndexBuffer {
	public:
		DXIndexBuffer(std::vector<uint32_t> indices);
		~DXIndexBuffer();

		int numCubeIndices;

		ID3D12Resource* indexBuffer; // a default buffer in GPU memory that we will load index data for our triangle into

		D3D12_INDEX_BUFFER_VIEW indexBufferView; // a structure holding information about the index buffer

		int iBufferSize;
	};
}