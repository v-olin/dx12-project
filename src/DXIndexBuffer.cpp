#include "DXIndexBuffer.h"
#include <DirectXMath.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <d3d12.h>
#include "../vendor/d3dx12/d3dx12.h"
#include "DXRenderer.h"
#include "Logger.h"

namespace pathtracex {
	DXIndexBuffer::DXIndexBuffer(std::vector<uint32_t> indices)
	{
		DXRenderer* renderer = DXRenderer::getInstance();

		renderer->resetCommandList();

		iBufferSize = sizeof(uint32_t) * indices.size();

		numCubeIndices = indices.size();
	//	renderer->resetCommandList();

		HRESULT hr;
		// create default heap to hold index buffer
		CD3DX12_HEAP_PROPERTIES iHeapProperties(D3D12_HEAP_TYPE_DEFAULT);
		CD3DX12_RESOURCE_DESC iResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(iBufferSize);
		hr = renderer->device->CreateCommittedResource(
			&iHeapProperties, // a default heap
			D3D12_HEAP_FLAG_NONE, // no flags
			&iResourceDesc, // resource description for a buffer
			D3D12_RESOURCE_STATE_COPY_DEST, // start in the copy destination state
			nullptr, // optimized clear value must be null for this type of resource
			IID_PPV_ARGS(&indexBuffer));

		if (FAILED(hr))
		{
			LOG_ERROR("FAILED TO CREATE INDEX BUFFER");
		}

		// we can give resource heaps a name so when we debug with the graphics debugger we know what resource we are looking at
		indexBuffer->SetName(L"Index Buffer Resource Heap");

		// create upload heap to upload index buffer
		CD3DX12_HEAP_PROPERTIES iHeapPropertiesUpload(D3D12_HEAP_TYPE_UPLOAD);
		CD3DX12_RESOURCE_DESC iResourceDescUpload = CD3DX12_RESOURCE_DESC::Buffer(iBufferSize);
		ID3D12Resource* iBufferUploadHeap;
		renderer->device->CreateCommittedResource(
			&iHeapPropertiesUpload, // upload heap
			D3D12_HEAP_FLAG_NONE, // no flags
			&iResourceDescUpload, // resource description for a buffer
			D3D12_RESOURCE_STATE_GENERIC_READ, // GPU will read from this buffer and copy its contents to the default heap
			nullptr,
			IID_PPV_ARGS(&iBufferUploadHeap));
		//		vBufferUploadHeap->SetName(L"Index Buffer Upload Resource Heap");

				// store vertex buffer in upload heap
		D3D12_SUBRESOURCE_DATA indexData = {};
		indexData.pData = reinterpret_cast<BYTE*>(&indices[0]); // pointer to our index array
		indexData.RowPitch = iBufferSize; // size of all our index buffer
		indexData.SlicePitch = iBufferSize; // also the size of our index buffer

		// we are now creating a command with the command list to copy the data from
		// the upload heap to the default heap
		UpdateSubresources(renderer->commandList, indexBuffer, iBufferUploadHeap, 0, 0, 1, &indexData);

		// transition the vertex buffer data from copy destination state to vertex buffer state
		CD3DX12_RESOURCE_BARRIER indexBufferResourceBarrier = CD3DX12_RESOURCE_BARRIER::Transition(indexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
		renderer->commandList->ResourceBarrier(1, &indexBufferResourceBarrier);

		// create a vertex buffer view for the triangle. We get the GPU memory address to the vertex pointer using the GetGPUVirtualAddress() method
		indexBufferView.BufferLocation = indexBuffer->GetGPUVirtualAddress();
		indexBufferView.Format = DXGI_FORMAT_R32_UINT; // 32-bit unsigned integer (this is what a dword is, double word, a word is 2 bytes)
		indexBufferView.SizeInBytes = iBufferSize;


		renderer->finishedRecordingCommandList();
		renderer->executeCommandList();



		renderer->incrementFenceAndSignalCurrentFrame();

	}
	DXIndexBuffer::~DXIndexBuffer()
	{
		SAFE_RELEASE(indexBuffer);
	}
}

