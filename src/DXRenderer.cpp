#include "DXRenderer.h"

#include "Exceptions.h"
#include "Helper.h"

#include <DirectXMath.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
//#include <dxgidebug.h>
#include "backends/imgui_impl_dx12.h"

namespace wrl = Microsoft::WRL;
namespace dx = DirectX;

namespace pathtracex {

	DXRenderer::DXRenderer(HWND windowHandle, UINT width, UINT height) :
		windowHandle(windowHandle),
		width(width),
		height(height),
		useWarpDevice(false),
		frameIdx(0),
		viewport({ 0.0f, 0.0f, (float)(width), (float)(height), 0.0f, 1.0f }),
		scissorRect({ 0, 0, (LONG)(width), (LONG)(height) }),
		rtvDescriptorSize(0),
		aspectRatio(static_cast<float>(width) / static_cast<float>(height))
	{ }

	void DXRenderer::onInit() {
#ifdef _DEBUG
		createDebugController();
#endif
		createFactory();
		createDevice();
		createCommandQueue();
		createSwapChain();
		createDescriptorHeaps();
		createCommandAllocators();
		createRootSignature();
		createPipeline();
		createCommandList();
		createFencesAndEvents();
		createTestModel();

		// dis do be correct i think?
		ImGui_ImplDX12_Init(pDevice.Get(), FRAME_COUNT,
			DXGI_FORMAT_R8G8B8A8_UNORM, srvHeap.Get(),
			srvHeap.Get()->GetCPUDescriptorHandleForHeapStart(),
			srvHeap.Get()->GetGPUDescriptorHandleForHeapStart());
	}

	void DXRenderer::onUpdate() {

	}

	void DXRenderer::onRender() {
		// record all commands for gpu into command list
		populateCommandList();

		// exec command list
		ID3D12CommandList* ppCommandLists[] = { cmdList.Get() };
		cmdQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

		// present next frame
		HRESULT hr;
		THROW_IF_FAILED(pSwap->Present(1, 0)); // with vsync
		waitForPreviousFrame();
	}

	void DXRenderer::onDestroy() {
		// wait for gpu to be finished using cpu resources
		waitForPreviousFrame();
		ImGui_ImplDX12_Shutdown();
		// cleanup device
		destroyDevice();
	}

	void DXRenderer::destroyDevice() {
		if (pSwap) { // is ComPtr so no need to call release
			pSwap->SetFullscreenState(false, nullptr);
		}

		// cmdAllocator is comptr
		CloseHandle(fenceEvent);
		/*
#ifdef _DEBUG
		IDXGIDebug1* pDebug = nullptr;
		if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&pDebug)))) {
			pDebug->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_SUMMARY);
			pDebug->Release();
		}
#endif
		*/
	}

	ID3D12GraphicsCommandList* const DXRenderer::borrowCommandListPointer() const noexcept {
		return cmdList.Get();
	}
	
	void DXRenderer::initGraphicsAPI()
	{
		// TODO
	}

	void DXRenderer::setClearColor(const dx::XMFLOAT3& color)
	{
		// TODO
	}

	void DXRenderer::setCullFace(bool enabled)
	{
		// TODO
	}

	void DXRenderer::setDepthTest(bool enabled)
	{
		// TODO
	}

	void DXRenderer::setDrawTriangleOutline(bool enabled)
	{
		// TODO
	}

	void DXRenderer::setViewport(int x, int y, int width, int height)
	{
		// TODO
	}

	void DXRenderer::createTestModel() {
		HRESULT hr;
		
		{
			struct Vertex {
				dx::XMFLOAT3 pos;
				dx::XMFLOAT4 color;
			};

			/* 3D Test
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
			*/

			Vertex vertices[] = {
				{ { 0.0f, 0.25f * aspectRatio, 0.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
				{ { 0.25f, -0.25f * aspectRatio, 0.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } },
				{ { -0.25f, -0.25f * aspectRatio, 0.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } },
			};
			const UINT vertexBufferSize = sizeof(vertices);

			D3D12_HEAP_PROPERTIES hprops = getDefaultHeapProperties();
			D3D12_RESOURCE_DESC rdesc = getResourceDescriptionFromSize(vertexBufferSize);

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
	}

	void DXRenderer::populateCommandList() {
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

		D3D12_RESOURCE_BARRIER rbarr = transitionBarrierFromRenderTarget(renderTargets[frameIdx].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
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
		else { // if raytracing
			const float clearColor[] = { 0.4f, 0.2f, 0.0f, 1.0f };
			cmdList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
		}
		
		cmdList->SetDescriptorHeaps(1, srvHeap.GetAddressOf());
		ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmdList.Get());

		// after render indicate that back buffer will be presented
		rbarr = transitionBarrierFromRenderTarget(renderTargets[frameIdx].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
		cmdList->ResourceBarrier(1, &rbarr);

		THROW_IF_FAILED(cmdList->Close());
	}

	/*
	void DXRenderer::waitForPreviousFrame()
	{
	    HRESULT hr;
	
	    // swap the current rtv buffer index so we draw on the correct buffer
	    frameIdx = pSwap->GetCurrentBackBufferIndex();
	
	    // if the current fence value is still less than "fenceValue", then we know the GPU has not finished executing
	    // the command queue since it has not reached the "commandQueue->Signal(fence, fenceValue)" command
	    if (fences[frameIdx]->GetCompletedValue() < fenceValues[frameIdx])
	    {
	        // we have the fence create an event which is signaled once the fence's current value is "fenceValue"
	        hr = fences[frameIdx]->SetEventOnCompletion(fenceValues[frameIdx], fenceEvent);
	
	        // We will wait until the fence has triggered the event that it's current value has reached "fenceValue". once it's value
	        // has reached "fenceValue", we know the command queue has finished executing
	        WaitForSingleObject(fenceEvent, INFINITE);
	    }
	
	    // increment fenceValue for next frame
	    fenceValues[frameIdx]++;
	}
*/

	void DXRenderer::waitForPreviousFrame() {
		HRESULT hr;
		// WAITING FOR THE FRAME TO COMPLETE BEFORE CONTINUING IS NOT BEST PRACTICE.
		// This is code implemented as such for simplicity. The D3D12HelloFrameBuffering
		// sample illustrates how to use fences for efficient resource usage and to
		// maximize GPU utilization.

		// Signal and increment the fence value.
		const UINT64 oldFence = fenceValue;
		THROW_IF_FAILED(cmdQueue->Signal(fence.Get(), oldFence));
		fenceValue++;

		if (fence->GetCompletedValue() < oldFence) {
			THROW_IF_FAILED(fence->SetEventOnCompletion(oldFence, fenceEvent));
			WaitForSingleObject(fenceEvent, INFINITE);
		}

		frameIdx = pSwap->GetCurrentBackBufferIndex();
	}

	void DXRenderer::getHardwareAdapter(IDXGIFactory2* pFactory, IDXGIAdapter1** ppAdapter) {
		wrl::ComPtr<IDXGIAdapter1> adapter;
		*ppAdapter = nullptr;

		for (UINT adaptIdx = 0; DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adaptIdx, &adapter); ++adaptIdx) {
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
				continue;
			}

			//if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, __uuidof(ID3D12Device), nullptr))) {
			if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
				break;
			}
		}

		*ppAdapter = adapter.Detach();
	}

	void DXRenderer::createFactory()
	{
		HRESULT hr;
		THROW_IF_FAILED(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));
	}

	void DXRenderer::createDebugController()
	{
		wrl::ComPtr<ID3D12Debug> debugController;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
			debugController->EnableDebugLayer();

			dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
		}
	}

	void DXRenderer::createDevice()
	{
		HRESULT hr;
		if (useWarpDevice) {
			wrl::ComPtr<IDXGIAdapter> warpAdapter;
			THROW_IF_FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));

			THROW_IF_FAILED(D3D12CreateDevice(
				warpAdapter.Get(),
				//D3D_FEATURE_LEVEL_12_1,
				D3D_FEATURE_LEVEL_11_0,
				IID_PPV_ARGS(&pDevice)
			));
		}
		else {
			wrl::ComPtr<IDXGIAdapter1> hardwareAdapter;
			getHardwareAdapter(factory.Get(), &hardwareAdapter);

			THROW_IF_FAILED(D3D12CreateDevice(
				hardwareAdapter.Get(),
				//D3D_FEATURE_LEVEL_12_1,
				D3D_FEATURE_LEVEL_11_0,
				IID_PPV_ARGS(&pDevice)
			));
		}
	}

	void DXRenderer::createCommandQueue()
	{
		HRESULT hr;
		D3D12_COMMAND_QUEUE_DESC queueDesc{};
		queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

		THROW_IF_FAILED(pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&cmdQueue)));
	}

	void DXRenderer::createSwapChain()
	{
		HRESULT hr;
		DXGI_SWAP_CHAIN_DESC1 swapChainDesc{};
		ZeroMemory(&swapChainDesc, sizeof(swapChainDesc));
		swapChainDesc.BufferCount = FRAME_COUNT;
		swapChainDesc.Width = width;
		swapChainDesc.Height = height;
		swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
		swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapChainDesc.SampleDesc.Count = 1;
		swapChainDesc.SampleDesc.Quality = 0;
		swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
		swapChainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
		swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
		swapChainDesc.Stereo = FALSE;

		wrl::ComPtr<IDXGISwapChain1> swapChain;
		// this is fcked
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
		//frameIdx = pSwap->GetCurrentBackBufferIndex();
	}
	void DXRenderer::createDescriptorHeaps()
	{
		HRESULT hr;
		{
			D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
			rtvHeapDesc.NumDescriptors = FRAME_COUNT;
			rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
			rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
			rtvHeapDesc.NodeMask = 1u;

			THROW_IF_FAILED(pDevice->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap)));

			rtvDescriptorSize = pDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		}

		{
			D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());

			for (UINT i = 0; i < FRAME_COUNT; i++) {
				THROW_IF_FAILED(pSwap->GetBuffer(i, IID_PPV_ARGS(&renderTargets[i])));
				pDevice->CreateRenderTargetView(renderTargets[i].Get(), nullptr, rtvHandle);
				rtvHandle.ptr += rtvDescriptorSize;
			}
		}

		{
			D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc{};
			srvHeapDesc.NumDescriptors = 1;
			srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
			srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

			THROW_IF_FAILED(pDevice->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&srvHeap)));
		}
	}

	void DXRenderer::createCommandAllocators()
	{
		HRESULT hr;

		THROW_IF_FAILED(pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAllocator)));
	}

	void DXRenderer::createRootSignature()
	{
		HRESULT hr;
		{
			D3D12_ROOT_SIGNATURE_DESC rootSignDesc = getRootSignatureDesc();
			wrl::ComPtr<ID3DBlob> signature;
			wrl::ComPtr<ID3DBlob> error;
			THROW_IF_FAILED(D3D12SerializeRootSignature(&rootSignDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error));
			THROW_IF_FAILED(pDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&rootSignature)));
		}
	}

	void DXRenderer::createPipeline() {
		Microsoft::WRL::ComPtr<ID3DBlob> vShader;
		Microsoft::WRL::ComPtr<ID3DBlob> pShader;
		HRESULT hr;

		UINT compileFlags = 0u;
#ifdef _DEBUG
		compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
		THROW_IF_FAILED(D3DCompileFromFile(L"../../shaders/VertexShader.hlsl", nullptr, nullptr, "main", "vs_5_0", compileFlags, 0, &vShader, nullptr));
		THROW_IF_FAILED(D3DCompileFromFile(L"../../shaders/PixelShader.hlsl", nullptr, nullptr, "main", "ps_5_0", compileFlags, 0, &pShader, nullptr));

		D3D12_INPUT_ELEMENT_DESC ied[] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "COLOR", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
		};

		D3D12_GRAPHICS_PIPELINE_STATE_DESC psd = getPreparedPipeStateDesc();
		psd.InputLayout = { ied, std::size(ied) };
		psd.pRootSignature = rootSignature.Get();
		psd.VS = { vShader.Get()->GetBufferPointer(), vShader.Get()->GetBufferSize() };
		psd.PS = { pShader.Get()->GetBufferPointer(), pShader.Get()->GetBufferSize() };

		THROW_IF_FAILED(pDevice->CreateGraphicsPipelineState(&psd, IID_PPV_ARGS(&pipelineState)));
	}
	
	void DXRenderer::createCommandList()
	{
		HRESULT hr;
		THROW_IF_FAILED(pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAllocator.Get(), pipelineState.Get(), IID_PPV_ARGS(&cmdList)));
		// Command lists are created in the recording state, but there is nothing
		// to record yet. The main loop expects it to be closed, so close it now.
		THROW_IF_FAILED(cmdList->Close());
	}

	void DXRenderer::createFencesAndEvents()
	{
		HRESULT hr;
		// create fence and wait for gpu upload
		THROW_IF_FAILED(pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
		fenceValue = 1; // set the initial fence value to 1

		fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
		if (fenceEvent == nullptr) {
			THROW_IF_FAILED(HRESULT_FROM_WIN32(GetLastError()));
		}

		// wait for commandlist to execute
		waitForPreviousFrame();
	}
}