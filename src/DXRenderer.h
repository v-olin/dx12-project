#pragma once

#include "PathWin.h"
#include "Event.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include "Helper.h"

#include <string>
#include <vector>
#include <wrl.h>

#include "GraphicsAPI.h"
#include "Window.h"
#include "Vertex.h"

#include "DXVertexBuffer.h"
#include "DXIndexBuffer.h"

namespace pathtracex {
	const int frameBufferCount = 3;

	// this will only call release if an object exists (prevents exceptions calling release on non existant objects)
	#define SAFE_RELEASE(p) { if ( (p) ) { (p)->Release(); (p) = 0; } }

	struct PointLight
	{
		float4 position; // using float4 to avoid packing issues
	};

	class DXRenderer : public GraphicsAPI, public IEventListener {
	public:
		~DXRenderer() = default;
		DXRenderer(const DXRenderer&) = delete;
		DXRenderer& operator=(const DXRenderer&) = delete;
	
		bool init(Window* window); // initializes direct3d 12

		void initGraphicsAPI() override;
		void setClearColor(const dx::XMFLOAT3& color) override;
		void setCullFace(bool enabled) override;
		void setDepthTest(bool enabled) override;
		void setDrawTriangleOutline(bool enabled) override;
		void setViewport(int x, int y, int width, int height) override;
		GraphicsAPIType getGraphicsAPIType() override { return GraphicsAPIType::DirectX12; };

		void Render(RenderSettings& renderSettings, Scene& scene); // execute the command list
		void Update(RenderSettings& renderSettings); // update the game logic

		static DXRenderer* getInstance() {
			static DXRenderer instance;
			return &instance;
		}


		void createTextureBuffer(ID3D12Resource** textureBuffer, ID3D12DescriptorHeap** descriptorHeap, D3D12_RESOURCE_DESC* textureDesc, BYTE* imageData, int bytesPerRow);
		void createIndexBuffer(ID3D12Resource** buffer, D3D12_INDEX_BUFFER_VIEW* bufferView, UINT64 bufferSize, BYTE* indexData);
		void createVertexBuffer(ID3D12Resource** buffer, D3D12_VERTEX_BUFFER_VIEW* bufferView, UINT64 bufferSize, BYTE* vertexData);

		void Cleanup(); // release com ojects and clean up memory
		
		virtual void onEvent(Event& e) override;

	private:
		ID3D12GraphicsCommandList* commandList; // a command list we can record commands into, then execute them to render the frame
		ID3D12Device* device; // direct3d device
		ID3D12CommandQueue* commandQueue; // container for command lists
		
		DXRenderer();


		HWND hwnd;
		bool useWarpDevice = false; // ???
		bool resizeOnNextFrame = false;
		UINT resizedWidth = 0, resizedHeight = 0;

		Window* window = nullptr;

		void finishedRecordingCommandList();
		void executeCommandList();
		void resetCommandList();
		void WaitForPreviousFrame(); // wait until gpu is finished with command list
		void incrementFenceAndSignalCurrentFrame();
		// direct3d stuff
 // number of buffers we want, 2 for double buffering, 3 for tripple buffering



		IDXGISwapChain3* swapChain; // swapchain used to switch between render targets



		ID3D12DescriptorHeap* rtvDescriptorHeap; // a descriptor heap to hold resources like the render targets

		ID3D12DescriptorHeap* srvHeap;

		ID3D12Resource* renderTargets[frameBufferCount]; // number of render targets equal to buffer count

		ID3D12CommandAllocator* commandAllocator[frameBufferCount]; // we want enough allocators for each buffer * number of threads (we only have one thread)



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

		// the total size of the buffer, and the size of each element (vertex)
		DXGI_SAMPLE_DESC sampleDesc{};

		ID3D12Resource* depthStencilBuffer; // This is the memory for our depth buffer. it will also be used for a stencil buffer in a later tutorial
		ID3D12DescriptorHeap* dsDescriptorHeap; // This is a heap for our depth/stencil buffer descriptor

		ID3D12DescriptorHeap* mainDescriptorHeap[frameBufferCount]; // this heap will store the descripor to our constant buffer
		ID3D12Resource* constantBufferUploadHeap[frameBufferCount]; // this is the memory on the gpu where our constant buffer will be placed.

		UINT8* cbColorMultiplierGPUAddress[frameBufferCount]; // this is a pointer to the memory location we get when we map our constant buffer



		// The constant buffer can't be bigger than 256 bytes
		struct ConstantBufferPerObject {
			DirectX::XMFLOAT4X4 wvpMat; // 64 bytes
			DirectX::XMFLOAT4X4 modelMatrix; // 64 bytes
			DirectX::XMFLOAT4X4 normalMatrix; // 64 bytes
			PointLight pointLights[3]; // 48 bytes
			int pointLightCount; // 4 bytes
			bool hasTexCoord; // 1 bytes (i think)
			// Total: 245 with hasTexCoord
		};

		// Constant buffers must be 256-byte aligned which has to do with constant reads on the GPU.
		// We are only able to read at 256 byte intervals from the start of a resource heap, so we will
		// make sure that we add padding between the two constant buffers in the heap (one for cube1 and one for cube2)
		// Another way to do this would be to add a float array in the constant buffer structure for padding. In this case
		// we would need to add a float padding[50]; after the wvpMat variable. This would align our structure to 256 bytes (4 bytes per float)
		// The reason i didn't go with this way, was because there would actually be wasted cpu cycles when memcpy our constant
		// buffer data to the gpu virtual address. currently we memcpy the size of our structure, which is 16 bytes here, but if we
		// were to add the padding array, we would memcpy 64 bytes if we memcpy the size of our structure, which is 50 wasted bytes
		// being copied.
		int ConstantBufferPerObjectAlignedSize = (sizeof(ConstantBufferPerObject) + 255) & ~255;

		ConstantBufferPerObject cbPerObject; // this is the constant buffer data we will send to the gpu 
		// (which will be placed in the resource we created above)

		ID3D12Resource* constantBufferUploadHeaps[frameBufferCount]; // this is the memory on the gpu where constant buffers for each frame will be placed

		UINT8* cbvGPUAddress[frameBufferCount]; // this is a pointer to each of the constant buffer resource heaps

		void UpdatePipeline(RenderSettings& renderSettings, Scene& scene); // update the direct3d pipeline (update command lists)


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
		bool createBuffers(bool createDepthBufferOnly = false);

		bool onWindowResizeEvent(WindowResizeEvent& wre);
		void onResizeUpdatePipeline();
		void onResizeUpdateRenderTargets();
		void onResizeUpdateBackBuffers();
		void onResizeUpdateDescriptorHeaps();
		void waitForTotalGPUCompletion();

		void destroyDevice();
	};

}