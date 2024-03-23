#pragma once

#include "PathWin.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include "Helper.h"

#include <string>
#include <vector>
#include <wrl.h>

#include "GraphicsAPI.h"
#include "Window.h"

namespace pathtracex {
	const int frameBufferCount = 3;

	struct Vertex {
		DirectX::XMFLOAT3 pos;
	};

	// this will only call release if an object exists (prevents exceptions calling release on non existant objects)
	#define SAFE_RELEASE(p) { if ( (p) ) { (p)->Release(); (p) = 0; } }

	class DXRenderer : GraphicsAPI {
	public:
		DXRenderer(Window& window);
		~DXRenderer() = default;
		DXRenderer(const DXRenderer&) = delete;
		DXRenderer& operator=(const DXRenderer&) = delete;
	
		bool InitD3D(); // initializes direct3d 12

		void initGraphicsAPI() override;
		void setClearColor(const dx::XMFLOAT3& color) override;
		void setCullFace(bool enabled) override;
		void setDepthTest(bool enabled) override;
		void setDrawTriangleOutline(bool enabled) override;
		void setViewport(int x, int y, int width, int height) override;
		GraphicsAPIType getGraphicsAPIType() override { return GraphicsAPIType::DirectX12; };

		void Render(); // execute the command list

	private:
		HWND hwnd;
		bool useWarpDevice; // ???

	private:
		Window& window;

		// direct3d stuff
 // number of buffers we want, 2 for double buffering, 3 for tripple buffering

		ID3D12Device* device; // direct3d device

		IDXGISwapChain3* swapChain; // swapchain used to switch between render targets

		ID3D12CommandQueue* commandQueue; // container for command lists

		ID3D12DescriptorHeap* rtvDescriptorHeap; // a descriptor heap to hold resources like the render targets

		ID3D12Resource* renderTargets[frameBufferCount]; // number of render targets equal to buffer count

		ID3D12CommandAllocator* commandAllocator[frameBufferCount]; // we want enough allocators for each buffer * number of threads (we only have one thread)

		ID3D12GraphicsCommandList* commandList; // a command list we can record commands into, then execute them to render the frame

		ID3D12Fence* fence[frameBufferCount];    // an object that is locked while our command list is being executed by the gpu. We need as many 
		//as we have allocators (more if we want to know when the gpu is finished with an asset)

		HANDLE fenceEvent; // a handle to an event when our fence is unlocked by the gpu

		UINT64 fenceValue[frameBufferCount]; // this value is incremented each frame. each fence will have its own value

		int frameIndex; // current rtv we are on

		int rtvDescriptorSize; // size of the rtv descriptor on the device (all front and back buffers will be the same size)

		IDXGIFactory4* dxgiFactory;

		ID3D12PipelineState* pipelineStateObject; // pso containing a pipeline state

		ID3D12RootSignature* rootSignature; // root signature defines data shaders will access

		D3D12_VIEWPORT viewport; // area that output from rasterizer will be stretched to.

		D3D12_RECT scissorRect; // the area to draw in. pixels outside that area will not be drawn onto

		ID3D12Resource* vertexBuffer; // a default buffer in GPU memory that we will load vertex data for our triangle into

		D3D12_VERTEX_BUFFER_VIEW vertexBufferView; // a structure containing a pointer to the vertex data in gpu memory
		// the total size of the buffer, and the size of each element (vertex)
		DXGI_SAMPLE_DESC sampleDesc{};


		// function declarations


		void Update(); // update the game logic

		void UpdatePipeline(); // update the direct3d pipeline (update command lists)



		void Cleanup(); // release com ojects and clean up memory

		void WaitForPreviousFrame(); // wait until gpu is finished with command list



		bool createFactory();
		bool createDebugController();
		bool createDevice();
		bool createCommandQueue();
		bool createSwapChain();
		bool createDescriptorHeaps();
		bool createCommandAllocators();
		bool createRootSignature();
		bool createPipeline();
		bool createCommandList();
		bool createFencesAndEvents();
		bool createVertexBuffer();

		void destroyDevice();
	};

}