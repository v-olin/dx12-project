#pragma once
#include <vector>
#include "Vertex.h"
#include <DirectXMath.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <d3d12.h>

namespace pathtracex {
	class DXVertexBuffer {
	public:
		DXVertexBuffer(std::vector<Vertex> vertices);
		~DXVertexBuffer();
		void bind();

		ID3D12Resource* vertexBuffer = nullptr; // a default buffer in GPU memory that we will load vertex data for our triangle into

		D3D12_VERTEX_BUFFER_VIEW vertexBufferView{}; // a structure containing a pointer to the vertex data in gpu memory

		size_t vBufferSize;
	};
}