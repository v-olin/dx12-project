#include "Renderer.h"

#include "Helper.h"

#include <DirectXMath.h>
#include <d3dcompiler.h>

#pragma comment(lib,"d3d12.lib")
#pragma comment(lib,"D3DCompiler.lib") // shader compiler

namespace wrl = Microsoft::WRL;
namespace dx = DirectX;

namespace pathtracex {

	Renderer::Renderer(HWND windowHandle, UINT width, UINT height) :
		windowHandle(windowHandle),
		width(width),
		height(height),
		useWarpDevice(false),
		frameIdx(0),
		viewport({ 0.0f, 0.0f, (float)(width), (float)(height), 0.0f, 1.0f }),
		scissorRect({ 0, 0, (LONG)(width), (LONG)(height) }),
		rtvDescriptorSize(0)
	{ }

	void Renderer::onInit() {
		loadPipeline();
		loadShaders();
	}

	void Renderer::loadPipeline() {
		HRESULT hr;
		UINT dxgiFactoryFlags = 0u;

#ifdef _DEBUG
		{
			wrl::ComPtr<ID3D12Debug> debugController;
			if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
				debugController->EnableDebugLayer();

				dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
			}
		}
#endif

		wrl::ComPtr<IDXGIFactory4> factory;
		THROW_IF_FAILED(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

		if (useWarpDevice) {
			wrl::ComPtr<IDXGIAdapter> warpAdapter;
			THROW_IF_FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));

			THROW_IF_FAILED(D3D12CreateDevice(
				warpAdapter.Get(),
				D3D_FEATURE_LEVEL_12_1,
				IID_PPV_ARGS(&pDevice)
			));
		}
		else {
			wrl::ComPtr<IDXGIAdapter1> hardwareAdapter;
			getHardwareAdapter(factory.Get(), &hardwareAdapter);

			THROW_IF_FAILED(D3D12CreateDevice(
				hardwareAdapter.Get(),
				D3D_FEATURE_LEVEL_12_1,
				IID_PPV_ARGS(&pDevice)
			));
		}

		D3D12_COMMAND_QUEUE_DESC queueDesc{};
		queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

		THROW_IF_FAILED(pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&cmdQueue)));

		DXGI_SWAP_CHAIN_DESC1 swapChainDesc{};
		swapChainDesc.BufferCount = FRAME_COUNT;
		swapChainDesc.Width = width;
		swapChainDesc.Height = height;
		swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapChainDesc.SampleDesc.Count = 1;

		wrl::ComPtr<IDXGISwapChain1> swapChain;
		THROW_IF_FAILED(factory->CreateSwapChainForHwnd(
			cmdQueue.Get(),
			windowHandle,
			&swapChainDesc,
			nullptr,
			nullptr,
			&swapChain
		));

		THROW_IF_FAILED(factory->MakeWindowAssociation(windowHandle, DXGI_MWA_NO_ALT_ENTER));

		THROW_IF_FAILED(swapChain.As(&pSwap));
		frameIdx = pSwap->GetCurrentBackBufferIndex();

		{
			D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
			rtvHeapDesc.NumDescriptors = FRAME_COUNT;
			rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
			rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
			
			THROW_IF_FAILED(pDevice->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap)));

			rtvDescriptorSize = pDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		}

		{
			D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());

			for (UINT i = 0; i < FRAME_COUNT; i++) {
				THROW_IF_FAILED(pSwap->GetBuffer(i, IID_PPV_ARGS(&renderTargets[i])));
				pDevice->CreateRenderTargetView(renderTargets[i].Get(), nullptr, rtvHandle);
				rtvHandle.ptr += 1 * rtvDescriptorSize;
			}
		}

		THROW_IF_FAILED(pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAllocator)));
	}

	void Renderer::loadShaders() {
		HRESULT hr;
		{
			D3D12_ROOT_SIGNATURE_DESC rootSignDesc{};
			rootSignDesc.NumParameters = 0;
			rootSignDesc.pParameters = nullptr;
			rootSignDesc.NumStaticSamplers = 0;
			rootSignDesc.pStaticSamplers = nullptr;
			rootSignDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

			wrl::ComPtr<ID3DBlob> signature;
			wrl::ComPtr<ID3DBlob> error;
			THROW_IF_FAILED(D3D12SerializeRootSignature(&rootSignDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error));
			THROW_IF_FAILED(pDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&rootSignature)));
		}

		{
			wrl::ComPtr<ID3DBlob> vShader;
			wrl::ComPtr<ID3DBlob> pShader;

			UINT compileFlags = 0u;
#ifdef _DEBUG
			compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
			THROW_IF_FAILED(D3DCompileFromFile(L"shader.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", compileFlags, 0, &vShader, nullptr));
			THROW_IF_FAILED(D3DCompileFromFile(L"shader.hlsl", nullptr, nullptr, "PSMain", "vs_5_0", compileFlags, 0, &pShader, nullptr));

			D3D12_INPUT_ELEMENT_DESC ied[] = {
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
			};

			D3D12_GRAPHICS_PIPELINE_STATE_DESC psd{};
			psd.InputLayout = { ied, std::size(ied) };
			psd.pRootSignature = rootSignature.Get();
			psd.VS = { vShader.Get()->GetBufferPointer(), vShader.Get()->GetBufferSize() };
			psd.PS = { pShader.Get()->GetBufferPointer(), pShader.Get()->GetBufferSize() };
			/*
			typedef struct D3D12_RASTERIZER_DESC
			{
			D3D12_FILL_MODE FillMode;
			D3D12_CULL_MODE CullMode;
			BOOL FrontCounterClockwise;
			INT DepthBias;
			FLOAT DepthBiasClamp;
			FLOAT SlopeScaledDepthBias;
			BOOL DepthClipEnable;
			BOOL MultisampleEnable;
			BOOL AntialiasedLineEnable;
			UINT ForcedSampleCount;
			D3D12_CONSERVATIVE_RASTERIZATION_MODE ConservativeRaster;
			} 	D3D12_RASTERIZER_DESC;
			*/
			psd.RasterizerState = {
				D3D12_FILL_MODE_SOLID,
				D3D12_CULL_MODE_BACK,
				FALSE,
				D3D12_DEFAULT_DEPTH_BIAS,
				D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
				D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
				TRUE, FALSE, FALSE,
				0, D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
			};
			/*
			typedef struct D3D12_BLEND_DESC
			{
			BOOL AlphaToCoverageEnable;
			BOOL IndependentBlendEnable;
			D3D12_RENDER_TARGET_BLEND_DESC RenderTarget[ 8 ];
			}
			*/
			D3D12_RENDER_TARGET_BLEND_DESC drtbd = {
				FALSE, FALSE,
				D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
				D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
				D3D12_LOGIC_OP_NOOP,
				D3D12_COLOR_WRITE_ENABLE_ALL,
			};
			D3D12_BLEND_DESC blendd = { FALSE, FALSE, drtbd };
			for (UINT i = 0; i < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) {
				blendd.RenderTarget[i] = drtbd;
			}
			psd.BlendState = blendd;
			psd.DepthStencilState.DepthEnable = FALSE;
			psd.DepthStencilState.StencilEnable = FALSE;
			psd.SampleMask = UINT_MAX;
			psd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			psd.NumRenderTargets = 1;
			psd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
			psd.SampleDesc.Count = 1;
			THROW_IF_FAILED(pDevice->CreateGraphicsPipelineState(&psd, IID_PPV_ARGS(&pipelineState)));
		}

		THROW_IF_FAILED(pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAllocator.Get(), pipelineState.Get(), IID_PPV_ARGS(&cmdList)));
		// Command lists are created in the recording state, but there is nothing
		// to record yet. The main loop expects it to be closed, so close it now.
		THROW_IF_FAILED(cmdList->Close());
		
		{
			struct Vertex {
				dx::XMFLOAT3 pos;
				dx::XMFLOAT4 color;
			};

			struct ConstantBuffer {
				dx::XMMATRIX transform;
			};

			// clockwise is correct order in D3D11!!!
			const Vertex vertices[]{
				// position					color
				{ {-1.0f, -1.0f, -1.0f },	{ 255, 0, 0, 0 } },
				{ {1.0f, -1.0f, -1.0f },	{ 0, 255, 0, 0 } },
				{ {-1.0f, 1.0f, -1.0f },	{ 0, 0, 255, 0 } },
				{ {1.0f, 1.0f, -1.0f },		{ 255, 0, 0, 0 } },
				{ {-1.0f, -1.0f, 1.0f },	{ 0, 255, 0, 0 } },
				{ {1.0f, -1.0f, 1.0f },		{ 0, 0, 255, 0 } },
				{ {-1.0f, 1.0f, 1.0f },		{ 255, 0, 0, 0 } },
				{ {1.0f, 1.0f, 1.0f },		{ 0, 255, 0, 0 } }
			};
			const UINT vertexBufferSize = sizeof(vertices);


			D3D12_HEAP_PROPERTIES hprops{};
			hprops.Type = D3D12_HEAP_TYPE_UPLOAD;
			hprops.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
			hprops.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
			hprops.CreationNodeMask = 1u;
			hprops.VisibleNodeMask = 1u;

			D3D12_RESOURCE_DESC rdesc{};
			rdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			rdesc.Alignment = 0ui64;
			rdesc.Width = vertexBufferSize;
			rdesc.Height = 1u;
			rdesc.DepthOrArraySize = 1u;
			rdesc.MipLevels = 1u;
			rdesc.Format = DXGI_FORMAT_UNKNOWN;
			rdesc.SampleDesc.Count = 1u;
			rdesc.SampleDesc.Quality = 0u;
			rdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			rdesc.Flags = D3D12_RESOURCE_FLAG_NONE;
			// Note: using upload heaps to transfer static data like vert buffers is not 
			// recommended. Every time the GPU needs it, the upload heap will be marshalled 
			// over. Please read up on Default Heap usage. An upload heap is used here for 
			// code simplicity and because there are very few verts to actually transfer.
			THROW_IF_FAILED(pDevice->CreateCommittedResource(
				&hprops, D3D12_HEAP_FLAG_NONE,
				&rdesc, D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&vertexBuffer)
			));

			// copy to buffer
			UINT8* pVertexDataBegin;
			D3D12_RANGE readRange(0, 0);
			THROW_IF_FAILED(vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
			memcpy(pVertexDataBegin, vertices, sizeof(vertices));
			vertexBuffer->Unmap(0, nullptr);

			// initialize buffer view
			vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
			vertexBufferView.StrideInBytes = sizeof(Vertex);
			vertexBufferView.SizeInBytes = vertexBufferSize;
		}

		// create fence and wait for gpu upload
		{
			THROW_IF_FAILED(pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
			fenceValue = 1ui64;
			
			fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
			if (fenceEvent == nullptr) {
				THROW_IF_FAILED(HRESULT_FROM_WIN32(GetLastError()));
			}

			// wait for commandlist to execute
			waitForPreviousFrame();
		}
	}

	void Renderer::populateCommandList() {
		HRESULT hr;

		// Command list allocators can only be reset when the associated 
		// command lists have finished execution on the GPU; apps should use 
		// fences to determine GPU execution progress.
		THROW_IF_FAILED(cmdAllocator->Reset());

		// However, when ExecuteCommandList() is called on a particular command 
		// list, that command list can then be reset at any time and must be before 
		// re-recording.
		THROW_IF_FAILED(cmdList->Reset(cmdAllocator.Get(), pipelineState.Get()));

		// base state
		cmdList->SetGraphicsRootSignature(rootSignature.Get());
		cmdList->RSSetViewports(1, &viewport);
		cmdList->RSSetScissorRects(1, &scissorRect);

		D3D12_RESOURCE_BARRIER rbarr{};
		ZeroMemory(&rbarr, sizeof(rbarr));
		rbarr.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		rbarr.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		rbarr.Transition.pResource = renderTargets[frameIdx].Get();
		rbarr.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
		rbarr.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
		rbarr.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		// indicate back buffer will be used as render target
		cmdList->ResourceBarrier(1, &rbarr);

		D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
		D3D12_CPU_DESCRIPTOR_HANDLE heapHandle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
		rtvHandle.ptr = heapHandle.ptr + frameIdx * rtvDescriptorSize;

		cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

		if (renderRasterized) {
			const float clearColor[] = { 0.0f, 0.2f, 0.4f, 1.0f };
			cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			cmdList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
			cmdList->IASetVertexBuffers(0, 1, &vertexBufferView);
			cmdList->DrawInstanced(3, 1, 0, 0);
		}
		else {

		}
	}

	void Renderer::getHardwareAdapter(IDXGIFactory2* pFactory, IDXGIAdapter1** ppAdapter) {
		wrl::ComPtr<IDXGIAdapter1> adapter;
		*ppAdapter = nullptr;

		for (UINT adaptIdx = 0; DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adaptIdx, &adapter); ++adaptIdx) {
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
				continue;
			}

			if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, __uuidof(ID3D12Device), nullptr))) {
				break;
			}
		}

		*ppAdapter = adapter.Detach();
	}
}