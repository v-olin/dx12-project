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
		float3 pos;
		float4 color;
	};

	class DXRenderer : GraphicsAPI {
	public:
		DXRenderer(Window& window);
		~DXRenderer() = default;
		DXRenderer(const DXRenderer&) = delete;
		DXRenderer& operator=(const DXRenderer&) = delete;
	


		void initGraphicsAPI() override;
		void setClearColor(const dx::XMFLOAT3& color) override;
		void setCullFace(bool enabled) override;
		void setDepthTest(bool enabled) override;
		void setDrawTriangleOutline(bool enabled) override;
		void setViewport(int x, int y, int width, int height) override;
		GraphicsAPIType getGraphicsAPIType() override { return GraphicsAPIType::DirectX12; };

	private:
		HWND windowHandle;
		bool useWarpDevice; // ???

	private:
		
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

		// function declarations
		bool InitD3D(); // initializes direct3d 12

		void Update(); // update the game logic

		void UpdatePipeline(); // update the direct3d pipeline (update command lists)

		void Render(); // execute the command list

		void Cleanup(); // release com ojects and clean up memory

		void WaitForPreviousFrame(); // wait until gpu is finished with command list



		void createFactory();
		void createDebugController();
		void createDevice();
		void createCommandQueue();
		void createSwapChain();
		void createDescriptorHeaps();
		void createCommandAllocators();
		void createRootSignature();
		void createPipeline();
		void createCommandList();
		void createFencesAndEvents();

		void destroyDevice();
	};

}