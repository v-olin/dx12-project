#include "DXVertexBuffer.h"
#include <DirectXMath.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <d3d12.h>
#include "../vendor/d3dx12/d3dx12.h"
#include "DXRenderer.h"

namespace pathtracex {


	DXVertexBuffer::DXVertexBuffer(std::vector<Vertex> vertices)
	{
		DXRenderer* renderer = DXRenderer::getInstance();

		vBufferSize = sizeof(Vertex) * vertices.size();

		BYTE* vertexData = reinterpret_cast<BYTE*>(vertices.data());
		renderer->createVertexBuffer(&vertexBuffer, &vertexBufferView, vBufferSize, vertexData);
	}

	DXVertexBuffer::~DXVertexBuffer()
	{
		SAFE_RELEASE(vertexBuffer);
	}

	void pathtracex::DXVertexBuffer::bind()
	{
	}

}
