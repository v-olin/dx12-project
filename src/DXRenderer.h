#pragma once

#include "PathWin.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectXMath.h>

#include <string>
#include <vector>
#include <wrl.h>

#include "GraphicsAPI.h"

#define FRAME_COUNT 2u

namespace pathtracex {

	struct Vertex {
		DirectX::XMFLOAT3 pos;
		DirectX::XMFLOAT4 color;
	};

	class Renderer : GraphicsAPI {
	public:
		Renderer(HWND windowHandle, UINT width, UINT height);
		~Renderer() = default;
		Renderer(const Renderer&) = delete;
		Renderer& operator=(const Renderer&) = delete;
	
		void onInit();
		void onUpdate();
		void onRender();
		void onDestroy();

		ID3D12GraphicsCommandList* const borrowCommandListPointer() const noexcept;

		void initGraphicsAPI() override;
		void setClearColor(const dx::XMFLOAT3& color) override;
		void setCullFace(bool enabled) override;
		void setDepthTest(bool enabled) override;
		void setDrawTriangleOutline(bool enabled) override;
		void setViewport(int x, int y, int width, int height) override;
		GraphicsAPIType getGraphicsAPIType() override { return GraphicsAPIType::DirectX12; };

	private:
		UINT width, height;
		float aspectRatio;
		HWND windowHandle;
		bool useWarpDevice; // ???

	private:
		
		// gpu pipeline objects
		D3D12_VIEWPORT viewport;
		D3D12_RECT scissorRect;
		Microsoft::WRL::ComPtr<IDXGISwapChain3> pSwap;
		Microsoft::WRL::ComPtr<ID3D12Device> pDevice;
		Microsoft::WRL::ComPtr<ID3D12Resource> renderTargets[FRAME_COUNT];
		Microsoft::WRL::ComPtr<ID3D12CommandAllocator> cmdAllocator;
		Microsoft::WRL::ComPtr<ID3D12CommandQueue> cmdQueue;
		Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
		Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvHeap;
		Microsoft::WRL::ComPtr<ID3D12PipelineState> pipelineState;
		Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cmdList;
		UINT rtvDescriptorSize;

		// app objects
		Microsoft::WRL::ComPtr<ID3D12Resource> vertexBuffer;
		D3D12_VERTEX_BUFFER_VIEW vertexBufferView;

		// synch objects
		UINT frameIdx;
		HANDLE fenceEvent;
		Microsoft::WRL::ComPtr<ID3D12Fence> fence;
		UINT64 fenceValue;

		bool renderRasterized = true;

		void loadPipeline();
		void loadShaders();
		void populateCommandList();
		void waitForPreviousFrame();
		void toggleRenderMode();
		void getHardwareAdapter(IDXGIFactory2* pFactory, IDXGIAdapter1** ppAdapter);
	};

}