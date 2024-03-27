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

		renderer->resetCommandList();

	//	renderer->resetCommandList();


		// create default heap
		// default heap is memory on the GPU. Only the GPU has access to this memory
		// To get data into this heap, we will have to upload the data using
		// an upload heap
		CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_DEFAULT);
		CD3DX12_RESOURCE_DESC bufferResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(vBufferSize);
		renderer->device->CreateCommittedResource(
			&heapProperties, // a default heap
			D3D12_HEAP_FLAG_NONE, // no flags
			&bufferResourceDesc, // resource description for a buffer
			D3D12_RESOURCE_STATE_COPY_DEST, // we will start this heap in the copy destination state since we will copy data
			// from the upload heap to this heap
			nullptr, // optimized clear value must be null for this type of resource. used for render targets and depth/stencil buffers
			IID_PPV_ARGS(&vertexBuffer));

		// we can give resource heaps a name so when we debug with the graphics debugger we know what resource we are looking at
		vertexBuffer->SetName(L"Vertex Buffer Resource Heap");

		// create upload heap
		// upload heaps are used to upload data to the GPU. CPU can write to it, GPU can read from it
		// We will upload the vertex buffer using this heap to the default heap
		CD3DX12_HEAP_PROPERTIES heapPropertiesUpload(D3D12_HEAP_TYPE_UPLOAD);
		CD3DX12_RESOURCE_DESC bufferResourceDescUpload = CD3DX12_RESOURCE_DESC::Buffer(vBufferSize);
		ID3D12Resource* vBufferUploadHeap;
		renderer->device->CreateCommittedResource(
			&heapPropertiesUpload, // upload heap
			D3D12_HEAP_FLAG_NONE, // no flags
			&bufferResourceDescUpload, // resource description for a buffer
			D3D12_RESOURCE_STATE_GENERIC_READ, // GPU will read from this buffer and copy its contents to the default heap
			nullptr,
			IID_PPV_ARGS(&vBufferUploadHeap));
		vBufferUploadHeap->SetName(L"Vertex Buffer Upload Resource Heap");

		// store vertex buffer in upload heap
		D3D12_SUBRESOURCE_DATA vertexData = {};
		vertexData.pData = reinterpret_cast<BYTE*>(&vertices[0]); // pointer to our vertex array
		vertexData.RowPitch = vBufferSize; // size of all our triangle vertex data
		vertexData.SlicePitch = vBufferSize; // also the size of our triangle vertex data

		// we are now creating a command with the command list to copy the data from
		// the upload heap to the default heap
		UpdateSubresources(renderer->commandList, vertexBuffer, vBufferUploadHeap, 0, 0, 1, &vertexData);

		// transition the vertex buffer data from copy destination state to vertex buffer state
		CD3DX12_RESOURCE_BARRIER vertexBufferResourceBarrier = CD3DX12_RESOURCE_BARRIER::Transition(vertexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
		renderer->commandList->ResourceBarrier(1, &vertexBufferResourceBarrier);
		
		// create a vertex buffer view for the triangle. We get the GPU memory address to the vertex pointer using the GetGPUVirtualAddress() method
		vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
		vertexBufferView.StrideInBytes = sizeof(Vertex);
		vertexBufferView.SizeInBytes = vBufferSize;

		renderer->finishedRecordingCommandList();

		renderer->executeCommandList();

		// We make sure the index buffer is uploaded to the GPU before the renderer uses it
		renderer->incrementFenceAndSignalCurrentFrame();
	}

	DXVertexBuffer::~DXVertexBuffer()
	{
		SAFE_RELEASE(vertexBuffer);
	}

	void pathtracex::DXVertexBuffer::bind()
	{
	}

}
