#include "DXRenderer.h"

#include "App.h"
#include "Exceptions.h"
#include "Helper.h"

#include <DirectXMath.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <d3d12.h>
#include "../vendor/d3dx12/d3dx12.h"

#include "backends/imgui_impl_dx12.h"
#include <stdexcept>
#include "Logger.h"

namespace wrl = Microsoft::WRL;
namespace dx = DirectX;

namespace pathtracex {

#define IID_PPV_ARGS(ppType) __uuidof(**(ppType)), IID_PPV_ARGS_Helper(ppType)

	DXRenderer::DXRenderer() {}

	void DXRenderer::finishedRecordingCommandList()
	{
		HRESULT hr;
		hr = commandList->Close();
		if (FAILED(hr))
		{
			LOG_ERROR("Error executing command list, executeCommandList()");
			THROW_IF_FAILED(hr);
		}
	
	}

	void DXRenderer::executeCommandList()
	{
		
		ID3D12CommandList *ppCommandLists[] = {commandList};
		commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

		// This is maybe sus
	//	resetCommandList();
	}

	void DXRenderer::resetCommandList()
	{
		// We have to wait for the gpu to finish with the command allocator before we reset it
		WaitForPreviousFrame();

		HRESULT hr;
		hr = commandAllocator[frameIndex]->Reset();
		if (FAILED(hr))
		{
			LOG_ERROR("Error resetting command allocator, resetCommandList()");
			THROW_IF_FAILED(hr);
		}
		
		hr = commandList->Reset(commandAllocator[frameIndex], pipelineStateObject);
		if (FAILED(hr))
		{
			LOG_ERROR("Error resetting command list, resetCommandList()");
			THROW_IF_FAILED(hr);
		}
	}

	void DXRenderer::incrementFenceAndSignalCurrentFrame()
	{
		HRESULT hr;
		fenceValue[frameIndex]++;
		hr = commandQueue->Signal(fence[frameIndex], fenceValue[frameIndex]);
		if (FAILED(hr))
		{
			LOG_ERROR("Error signaling current frame and incrementing fence, signalCurrentFrameAndIncrementFence()");
			THROW_IF_FAILED(hr);
		}
	}

	void DXRenderer::onEvent(Event& e) {
		EventDispatcher dispatcher{ e };

		if (e.getEventType() == EventType::WindowResize) {
			dispatcher.dispatch<WindowResizeEvent>(BIND_EVENT_FN(DXRenderer::onWindowResizeEvent));
		}
	}

	bool DXRenderer::onWindowResizeEvent(WindowResizeEvent& wre) {
		/* OLD IMPL
		WaitForPreviousFrame();

		resetCommandList();

		auto backBuffIdx = swapChain->GetCurrentBackBufferIndex();
		for (int i = 0; i < frameBufferCount; i++) {
			renderTargets[i]->Release();
			renderTargets[i] = nullptr;
			//fenceValue[i] = fenceValue[backBuffIdx];
		}

		HRESULT hr;
		THROW_IF_FAILED(swapChain->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, 0u));

		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
		for (int i = 0; i < frameBufferCount; i++) {
			THROW_IF_FAILED(swapChain->GetBuffer(i, IID_PPV_ARGS(&renderTargets[i])));

			D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = {};
			rtvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;

			device->CreateRenderTargetView(renderTargets[i], nullptr, rtvHandle);
		}

		frameIndex = swapChain->GetCurrentBackBufferIndex();

		bool createOnlyDepthStencilBuffer = true;
		createBuffers(createOnlyDepthStencilBuffer); // create new stencil buffer

		viewport.Width = wre.getWidth();
		viewport.Height = wre.getHeight();
		viewport.MinDepth = D3D12_MIN_DEPTH;
		viewport.MaxDepth = D3D12_MAX_DEPTH;

		scissorRect.right = wre.getWidth();
		scissorRect.bottom = wre.getHeight();

		executeCommandList();
		incrementFenceAndSignalCurrentFrame();
		//THROW_IF_FAILED(swapChain->Present(0, 0));

		*/
		
		resizeOnNextFrame = true;
		resizedWidth = wre.getWidth();
		resizedHeight = wre.getHeight();
		
		return true;
	}

	bool DXRenderer::init(Window *window)
	{
		App::registerEventListener(this);

		LOG_INFO("Initializing DXRenderer");
		this->window = window;
		hwnd = window->windowHandle;

#ifdef _DEBUG
		if (!createDebugController())
			return false;
#endif
		if (!createFactory())
			return false;

		if (!createDevice())
			return false;

		if (!createCommandQueue())
			return false;

		if (!createSwapChain())
			return false;

		if (!createDescriptorHeaps())
			return false;

		if (!createCommandAllocators())
			return false;

		if (!createCommandList())
			return false;

		if (!createFencesAndEvents())
			return false;

		if (!createRootSignature())
			return false;

		if (!createPipeline())
			return false;

		if (!createBuffers())
			return false;

		int width, height;
		window->getSize(width, height);

		// Fill out the Viewport
		viewport.TopLeftX = 0;
		viewport.TopLeftY = 0;
		viewport.Width = width;
		viewport.Height = height;
		viewport.MinDepth = 0.0f;
		viewport.MaxDepth = 1.0f;

		// Fill out a scissor rect
		scissorRect.left = 0;
		scissorRect.top = 0;
		scissorRect.right = width;
		scissorRect.bottom = height;

		// dis do be correct i think?
		ImGui_ImplDX12_Init(device, frameBufferCount,
							DXGI_FORMAT_R8G8B8A8_UNORM, srvHeap,
							srvHeap->GetCPUDescriptorHandleForHeapStart(),
							srvHeap->GetGPUDescriptorHandleForHeapStart());

		LOG_INFO("DXRenderer initialized");
		return true;
	}

	void DXRenderer::initGraphicsAPI()
	{
		// TODO
	}

	void DXRenderer::setClearColor(const dx::XMFLOAT3 &color)
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

	void DXRenderer::onResizeUpdatePipeline() {
		resizeOnNextFrame = false;
		HRESULT hr;

		// step 1: yeet all the old framebuffers because they are of the old
		// resolution and worthless
		onResizeUpdateRenderTargets();

		// step 2: resize the whole fucking swapchain
		THROW_IF_FAILED(swapChain->ResizeBuffers(frameBufferCount, resizedWidth, resizedHeight, DXGI_FORMAT_UNKNOWN, 0u));
		frameIndex = swapChain->GetCurrentBackBufferIndex();

		// step 3: create new back buffers with the new size given by the updated swapchain
		onResizeUpdateBackBuffers();

		// step 4: update all heap bullshit for imgui
		onResizeUpdateDescriptorHeaps();

		// step 5: update viewport and scissor
		viewport.Width = resizedWidth;
		viewport.Height = resizedHeight;

		scissorRect.right = resizedWidth;
		scissorRect.bottom = resizedHeight;

		// step 6: reset imgui
		ImGui_ImplDX12_InvalidateDeviceObjects();
		ImGui_ImplDX12_CreateDeviceObjects();
	}

	void DXRenderer::onResizeUpdateRenderTargets() {
		auto bbi = swapChain->GetCurrentBackBufferIndex();
		for (int i = 0; i < frameBufferCount; i++) {
			renderTargets[i]->Release();
			renderTargets[i] = nullptr;
			fenceValue[i] = fenceValue[bbi];
		}
	}

	void DXRenderer::onResizeUpdateBackBuffers() {
		HRESULT hr;
		// create desc
		D3D12_DESCRIPTOR_HEAP_DESC rtvd{};
		ZeroMemory(&rtvd, sizeof(rtvd));
		rtvd.NumDescriptors = frameBufferCount;
		rtvd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
		rtvd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

		// release old heap and create new one
		rtvDescriptorHeap->Release();
		THROW_IF_FAILED(device->CreateDescriptorHeap(&rtvd, IID_PPV_ARGS(&rtvDescriptorHeap)));

		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

		D3D12_DESCRIPTOR_HEAP_DESC srvd{};
		ZeroMemory(&srvd, sizeof(srvd));
		srvd.NumDescriptors = 1;
		srvd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		srvd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

		srvHeap->Release();
		THROW_IF_FAILED(device->CreateDescriptorHeap(&srvd, IID_PPV_ARGS(&srvHeap)));

		rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		for (int i = 0; i < frameBufferCount; i++) {
			THROW_IF_FAILED(swapChain->GetBuffer(i, IID_PPV_ARGS(&renderTargets[i])));
			device->CreateRenderTargetView(renderTargets[i], nullptr, rtvHandle);
			rtvHandle.Offset(1, rtvDescriptorSize);
		}
	}

	void DXRenderer::onResizeUpdateDescriptorHeaps() {
		HRESULT hr;

		D3D12_DESCRIPTOR_HEAP_DESC dsvd{};
		ZeroMemory(&dsvd, sizeof(dsvd));
		dsvd.NumDescriptors = 1;
		dsvd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
		dsvd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

		dsDescriptorHeap->Release();
		THROW_IF_FAILED(device->CreateDescriptorHeap(&dsvd, IID_PPV_ARGS(&dsDescriptorHeap)));

		D3D12_DEPTH_STENCIL_VIEW_DESC dstvd{};
		ZeroMemory(&dstvd, sizeof(dstvd));
		dstvd.Format = DXGI_FORMAT_D32_FLOAT;
		dstvd.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
		dstvd.Flags = D3D12_DSV_FLAG_NONE;

		D3D12_CLEAR_VALUE depthOptimizedClearValue = {};
		depthOptimizedClearValue.Format = DXGI_FORMAT_D32_FLOAT;
		depthOptimizedClearValue.DepthStencil.Depth = 1.0f;
		depthOptimizedClearValue.DepthStencil.Stencil = 0;

		int Width, Height;
		window->getSize(Width, Height);

		CD3DX12_HEAP_PROPERTIES heapPropertiesDefault(D3D12_HEAP_TYPE_DEFAULT);
		CD3DX12_RESOURCE_DESC depthStencilResourceDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, Width, Height, 1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);

		depthStencilBuffer->Release();
		device->CreateCommittedResource(
			&heapPropertiesDefault,
			D3D12_HEAP_FLAG_NONE,
			&depthStencilResourceDesc,
			D3D12_RESOURCE_STATE_DEPTH_WRITE,
			&depthOptimizedClearValue,
			IID_PPV_ARGS(&depthStencilBuffer));
		dsDescriptorHeap->SetName(L"Depth/Stencil Resource Heap");

		device->CreateDepthStencilView(depthStencilBuffer, &dstvd, dsDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

		for (int i = 0; i < frameBufferCount; i++) {
			CD3DX12_HEAP_PROPERTIES heapUpload(D3D12_HEAP_TYPE_UPLOAD);
			CD3DX12_RESOURCE_DESC cbResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(1024 * 64); // allocate a 64KB buffer
			constantBufferUploadHeaps[i]->Release();
			hr = device->CreateCommittedResource(
				&heapUpload,					   // this heap will be used to upload the constant buffer data
				D3D12_HEAP_FLAG_NONE,			   // no flags
				&cbResourceDesc,				   // size of the resource heap. Must be a multiple of 64KB for single-textures and constant buffers
				D3D12_RESOURCE_STATE_GENERIC_READ, // will be data that is read from so we keep it in the generic read state
				nullptr,						   // we do not have use an optimized clear value for constant buffers
				IID_PPV_ARGS(&constantBufferUploadHeaps[i]));
			constantBufferUploadHeaps[i]->SetName(L"Constant Buffer Upload Resource Heap");

			ZeroMemory(&cbPerObject, sizeof(cbPerObject));

			CD3DX12_RANGE readRange(0, 0); // We do not intend to read from this resource on the CPU. (so end is less than or equal to begin)

			// map the resource heap to get a gpu virtual address to the beginning of the heap
			hr = constantBufferUploadHeaps[i]->Map(0, &readRange, reinterpret_cast<void**>(&cbvGPUAddress[i]));

			// Because of the constant read alignment requirements, constant buffer views must be 256 bit aligned. Our buffers are smaller than 256 bits,
			// so we need to add spacing between the two buffers, so that the second buffer starts at 256 bits from the beginning of the resource heap.
			memcpy(cbvGPUAddress[i], &cbPerObject, sizeof(cbPerObject));									  // cube1's constant buffer data
			memcpy(cbvGPUAddress[i] + ConstantBufferPerObjectAlignedSize, &cbPerObject, sizeof(cbPerObject)); // cube2's constant buffer data
		}
	}

	void DXRenderer::waitForTotalGPUCompletion() {
		for (int i = 0; i < frameBufferCount; i++) {
			UINT64 fenceValueForSignal = ++fenceValue[i];
			commandQueue->Signal(fence[i], fenceValueForSignal);
			if (fence[i]->GetCompletedValue() < fenceValue[i]) {
				fence[i]->SetEventOnCompletion(fenceValueForSignal, fenceEvent);
				WaitForSingleObject(fenceEvent, INFINITE);
			}
		}

		frameIndex = 0;
	}

	void DXRenderer::UpdatePipeline(RenderSettings &renderSettings, Scene &scene)
	{
		HRESULT hr;

		// We have to wait for the gpu to finish with the command allocator before we reset it
		if (resizeOnNextFrame) [[unlikely]] {
			waitForTotalGPUCompletion();
		}
		else {
			WaitForPreviousFrame();
		}


		// we can only reset an allocator once the gpu is done with it
		// resetting an allocator frees the memory that the command list was stored in
		hr = commandAllocator[frameIndex]->Reset();
		if (FAILED(hr)) {
			THROW_IF_FAILED(hr);
		}

		// reset the command list. by resetting the command list we are putting it into
		// a recording state so we can start recording commands into the command allocator.
		// the command allocator that we reference here may have multiple command lists
		// associated with it, but only one can be recording at any time. Make sure
		// that any other command lists associated to this command allocator are in
		// the closed state (not recording).
		// Here you will pass an initial pipeline state object as the second parameter,
		// but in this tutorial we are only clearing the rtv, and do not actually need
		// anything but an initial default pipeline, which is what we get by setting
		// the second parameter to NULL
		hr = commandList->Reset(commandAllocator[frameIndex], pipelineStateObject);
		if (FAILED(hr)) {
			THROW_IF_FAILED(hr);
		}


		// here we start recording commands into the commandList (which all the commands will be stored in the commandAllocator)

		if (resizeOnNextFrame) [[unlikely]] {
			onResizeUpdatePipeline();
		}


		// transition the "frameIndex" render target from the present state to the render target state so the command list draws to it starting from here
		CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
		commandList->ResourceBarrier(1, &barrier);

		// here we again get the handle to our current render target view so we can set it as the render target in the output merger stage of the pipeline
		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), frameIndex, rtvDescriptorSize);

		// get a handle to the depth/stencil buffer
		CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(dsDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

		// set the render target for the output merger stage (the output of the pipeline)
		commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

		// Clear the render target by using the ClearRenderTargetView command
		const float clearColor[] = {0.0f, 0.2f, 0.4f, 1.0f};
		commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
		commandList->ClearDepthStencilView(dsDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
		commandList->SetGraphicsRootSignature(rootSignature); // set the root signature

		// draw triangle
		commandList->RSSetViewports(1, &viewport);								  // set the viewports
		commandList->RSSetScissorRects(1, &scissorRect);						  // set the scissor rects
		commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST); // set the primitive topology

		int i = 0;
		std::vector<std::shared_ptr<Model>> models = scene.models;

		if (renderSettings.drawProcedualWorld) {
			// Add the procedual models to the list of models
			for (auto model : scene.proceduralGroundModels) {
				models.push_back(model);
			}

			for (auto model : scene.proceduralSkyModels) {
				models.push_back(model);
			}
		}

		int k = 0;
		PointLight pointLights[3];
		for (auto light : scene.lights) {
			pointLights[k] = { {light->transform.getPosition().x, light->transform.getPosition().y, light->transform.getPosition().z, 0} };
			k++;
		}

		cbPerObject.pointLightCount = k;

		// create the wvp matrix and store in constant buffer
		DirectX::XMMATRIX viewMat = renderSettings.camera.getViewMatrix();													// load view matrix
		DirectX::XMMATRIX projMat = renderSettings.camera.getProjectionMatrix(renderSettings.width, renderSettings.height); // load projection matrix

		for (auto model : models)
		{

			DirectX::XMMATRIX wvpMat = model->trans.transformMatrix * viewMat * projMat;										// create wvp matrix
			DirectX::XMMATRIX transposed = DirectX::XMMatrixTranspose(wvpMat);													// must transpose wvp matrix for the gpu
			DirectX::XMStoreFloat4x4(&cbPerObject.wvpMat, transposed);	// store transposed wvp matrix in constant buffer
			//DirectX::XMMATRIX modelMatrix = DirectX::XMMatrixTranspose(model->trans.transformMatrix);
			DirectX::XMMATRIX modelMatrix = DirectX::XMMatrixTranspose(model->trans.getModelMatrix());
			DirectX::XMMATRIX transposed2 = modelMatrix;
			DirectX::XMStoreFloat4x4(&cbPerObject.modelMatrix, transposed2);	// store the model matrix in the constant buffer
			DirectX::XMMATRIX normalMatrix = DirectX::XMMatrixTranspose(DirectX::XMMatrixInverse(nullptr, model->trans.transformMatrix));
			DirectX::XMStoreFloat4x4(&cbPerObject.normalMatrix, normalMatrix);
			int k = 0;
			PointLight pointLights[3];
			for (auto light : scene.lights) {
				pointLights[k] = { {light->transform.getPosition().x, light->transform.getPosition().y, light->transform.getPosition().z, 0}};
				k++;
			}

			cbPerObject.pointLightCount = k;

			memcpy(cbPerObject.pointLights, pointLights, sizeof(pointLights));

			
			// set cube1's constant buffer
			commandList->SetGraphicsRootConstantBufferView(0, constantBufferUploadHeaps[frameIndex]->GetGPUVirtualAddress() + ConstantBufferPerObjectAlignedSize * i);
			commandList->SetGraphicsRootSignature(rootSignature); // set the root signature

			// draw first cube
			commandList->IASetVertexBuffers(0, 1, &(model->vertexBuffer->vertexBufferView)); // set the vertex buffer (using the vertex buffer view)
			commandList->IASetIndexBuffer(&model->indexBuffer->indexBufferView);
			//commandList->DrawIndexedInstanced(model->indexBuffer->numCubeIndices, 1, 0, 0, 0);
			for (auto mesh : model->meshes) {
				//we need to add more uniforms so that we know if there are color textures and so on, 
				// all textures that are valid should be send down and used
				// all valid textures ARE sent to the GPU via the mainDescriptorHeap of the material
				// we just have to tell the shader what textures are valid
				cbPerObject.hasTexCoord = false;
				cbPerObject.hasNormalTex = false;
				cbPerObject.hasShinyTex = false;
				if (model->materials.size() > 0) {
					auto mat = model->materials[mesh.materialIdx];

					auto colTex = mat.colorTexture;
					auto normalTex = mat.normalTexture;
					auto shinyTex = mat.shininessTexture;

					cbPerObject.hasTexCoord = colTex.valid;
					cbPerObject.hasNormalTex = normalTex.valid;
					cbPerObject.hasShinyTex = shinyTex.valid;
			
					cbPerObject.material_shininess = mat.shininess;
					cbPerObject.material_metalness = mat.metalness;
					cbPerObject.material_fresnel = mat.fresnel;

					ID3D12DescriptorHeap* descriptorHeaps[] = { mat.mainDescriptorHeap };
					commandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
					commandList->SetGraphicsRootDescriptorTable(1, mat.mainDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
				}
				// // copy our ConstantBuffer instance to the mapped constant buffer resource
				//here also set all uniforms for each mesh
				memcpy(cbvGPUAddress[frameIndex] + ConstantBufferPerObjectAlignedSize * i, &cbPerObject, sizeof(cbPerObject));
				commandList->DrawIndexedInstanced(mesh.numberOfVertices, 1, 0, mesh.startIndex, 0);
			}

			i++;
		}

		commandList->SetDescriptorHeaps(1, &srvHeap);
		ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), commandList);

		// transition the "frameIndex" render target from the render target state to the present state. If the debug layer is enabled, you will receive a
		// warning if present is called on the render target when it's not in the present state
		CD3DX12_RESOURCE_BARRIER barrier2 = CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
		commandList->ResourceBarrier(1, &barrier2);

		hr = commandList->Close();
		if (FAILED(hr))
		{
		}
	}

	void DXRenderer::Render(RenderSettings &renderSettings, Scene &scene)
	{
		HRESULT hr;

		//	Update(renderSettings);
		UpdatePipeline(renderSettings, scene); // update the pipeline by sending commands to the commandqueue

		// create an array of command lists (only one command list here)
		ID3D12CommandList *ppCommandLists[] = {commandList};

		// execute the array of command lists
		commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

		// this command goes in at the end of our command queue. we will know when our command queue
		// has finished because the fence value will be set to "fenceValue" from the GPU since the command
		// queue is being executed on the GPU
		hr = commandQueue->Signal(fence[frameIndex], fenceValue[frameIndex]);
		if (FAILED(hr))
		{
			THROW_IF_FAILED(hr);
		}

		// present the current backbuffer
		hr = swapChain->Present(0, 0);
		if (FAILED(hr))
		{
			HRESULT reason = device->GetDeviceRemovedReason();
			THROW_IF_FAILED(reason);
		}
	}

	bool DXRenderer::createFactory()
	{
		LOG_TRACE("Creating DirectX12 factory");
		HRESULT hr;
		hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory));
		if (FAILED(hr))
		{
			LOG_FATAL("Error creating factory, createFactory()");
			return false;
		}
		LOG_TRACE("DirectX12 factory created");
		return true;
	}

	bool DXRenderer::createDebugController()
	{
#if defined(_DEBUG)
		// Always enable the debug layer before doing anything DX12 related
		// so all possible errors generated while creating DX12 objects
		// are caught by the debug layer.
		ID3D12Debug* debugInterface;
		D3D12GetDebugInterface(IID_PPV_ARGS(&debugInterface));
		debugInterface->EnableDebugLayer();
#endif
		return true;
	}

	bool DXRenderer::createDevice()
	{
		LOG_TRACE("Creating DirectX12 device");
		HRESULT hr;
		IDXGIAdapter1 *adapter; // adapters are the graphics card (this includes the embedded graphics on the motherboard)

		int adapterIndex = 0; // we'll start looking for directx 12  compatible graphics devices starting at index 0

		bool adapterFound = false; // set this to true when a good one was found

		// find first hardware gpu that supports d3d 12
		while (dxgiFactory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND)
		{
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
			{
				// we dont want a software device
				adapterIndex++; // add this line here. Its not currently in the downloadable project
				continue;
			}

			// we want a device that is compatible with direct3d 12 (feature level 11 or higher)
			hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr);
			if (SUCCEEDED(hr))
			{
				adapterFound = true;
				break;
			}

			adapterIndex++;
		}

		if (!adapterFound)
		{
			return false;
		}

		// Create the device
		hr = D3D12CreateDevice(
			adapter,
			D3D_FEATURE_LEVEL_11_0,
			IID_PPV_ARGS(&device));
		
		LOG_TRACE("DirectX12 device created");
		return !FAILED(hr);
	}

	bool DXRenderer::createCommandQueue()
	{
		LOG_TRACE("Creating DirectX12 command queue");
		HRESULT hr;
		// -- Create the Command Queue -- //

		D3D12_COMMAND_QUEUE_DESC cqDesc = {}; // we will be using all the default values
		cqDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		cqDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT; // direct means the gpu can directly execute this command queue

		hr = device->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&commandQueue)); // create the command queue

		LOG_TRACE("DirectX12 command queue created");
		return !FAILED(hr);
	}

	bool DXRenderer::createSwapChain()
	{
		LOG_TRACE("Creating DirectX12 swap chain");
		HRESULT hr;
		int width, height;
		window->getSize(width, height);

		// -- Create the Swap Chain (double/tripple buffering) -- //
		DXGI_MODE_DESC backBufferDesc = {};					// this is to describe our display mode
		backBufferDesc.Width = width;						// buffer width
		backBufferDesc.Height = height;						// buffer height
		backBufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // format of the buffer (rgba 32 bits, 8 bits for each chanel)

		// describe our multi-sampling. We are not multi-sampling, so we set the count to 1 (we need at least one sample of course)
		sampleDesc.Count = 1; // multisample count (no multisampling, so we just put 1, since we still need 1 sample)

		// Describe and create the swap chain.
		DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
		swapChainDesc.BufferCount = frameBufferCount;				 // number of buffers we have
		swapChainDesc.BufferDesc = backBufferDesc;					 // our back buffer description
		swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; // this says the pipeline will render to this swap chain
		swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;	 // dxgi will discard the buffer (data) after we call present
		swapChainDesc.OutputWindow = hwnd;							 // handle to our window
		swapChainDesc.SampleDesc = sampleDesc;						 // our multi-sampling description
		swapChainDesc.Windowed = true;								 // set to true, then if in fullscreen must call SetFullScreenState with true for full screen to get uncapped fps

		IDXGISwapChain *tempSwapChain;

		dxgiFactory->CreateSwapChain(
			commandQueue,	// the queue will be flushed once the swap chain is created
			&swapChainDesc, // give it the swap chain description we created above
			&tempSwapChain	// store the created swap chain in a temp IDXGISwapChain interface
		);

		swapChain = static_cast<IDXGISwapChain3 *>(tempSwapChain);

		frameIndex = swapChain->GetCurrentBackBufferIndex();

		LOG_TRACE("DirectX12 swap chain created");
		return true;
	}

	bool DXRenderer::createDescriptorHeaps()
	{
		LOG_TRACE("Creating DirectX12 descriptor heaps");

		HRESULT hr;
		// -- Create the Back Buffers (render target views) Descriptor Heap -- //

		// describe an rtv descriptor heap and create
		D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
		rtvHeapDesc.NumDescriptors = frameBufferCount;	   // number of descriptors for this heap.
		rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV; // this heap is a render target view heap

		// This heap will not be directly referenced by the shaders (not shader visible), as this will store the output from the pipeline
		// otherwise we would set the heap's flag to D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
		rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		hr = device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvDescriptorHeap));
		if (FAILED(hr))
		{
			LOG_FATAL("Error creating descriptor heap, createDescriptorHeaps()");
			return false;
		}

		// get the size of a descriptor in this heap (this is a rtv heap, so only rtv descriptors should be stored in it.
		// descriptor sizes may vary from device to device, which is why there is no set size and we must ask the
		// device to give us the size. we will use this size to increment a descriptor handle offset
		rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

		// get a handle to the first descriptor in the descriptor heap. a handle is basically a pointer,
		// but we cannot literally use it like a c++ pointer.
		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

		D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc{};
		srvHeapDesc.NumDescriptors = 1;
		srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

		hr = device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&srvHeap));
		if (FAILED(hr))
		{
			return false;
		}

		// Create a RTV for each buffer (double buffering is two buffers, tripple buffering is 3).
		for (int i = 0; i < frameBufferCount; i++)
		{
			// first we get the n'th buffer in the swap chain and store it in the n'th
			// position of our ID3D12Resource array
			hr = swapChain->GetBuffer(i, IID_PPV_ARGS(&renderTargets[i]));
			if (FAILED(hr))
			{
				LOG_FATAL("Error creating render target, createDescriptorHeaps()");
				return false;
			}

			// the we "create" a render target view which binds the swap chain buffer (ID3D12Resource[n]) to the rtv handle
			device->CreateRenderTargetView(renderTargets[i], nullptr, rtvHandle);

			// we increment the rtv handle by the rtv descriptor size we got above
			rtvHandle.Offset(1, rtvDescriptorSize);
		}
	
		LOG_TRACE("DirectX12 descriptor heaps created");

		return true;
	}

	bool DXRenderer::createCommandAllocators()
	{
		LOG_TRACE("Creating DirectX12 command allocators");

		HRESULT hr;
		// -- Create the Command Allocators -- //

		for (int i = 0; i < frameBufferCount; i++)
		{
			hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator[i]));
			if (FAILED(hr))
			{
				LOG_FATAL("Error creating command allocator, createCommandAllocators()");
				return false;
			}
		}

		LOG_TRACE("DirectX12 command allocators created");

		return true;
	}

	bool DXRenderer::createRootSignature()
	{
		LOG_TRACE("Creating DirectX12 root signature");

		HRESULT hr;
		// create a root descriptor, which explains where to find the data for this root parameter
		D3D12_ROOT_DESCRIPTOR rootCBVDescriptor;
		rootCBVDescriptor.RegisterSpace = 0;
		rootCBVDescriptor.ShaderRegister = 0;

		// create a descriptor range (descriptor table) and fill it out
		// this is a range of descriptors inside a descriptor heap
		CD3DX12_DESCRIPTOR_RANGE texTable;
		texTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, NUMTEXTURETYPES, 0);

		// create a root parameter and fill it out
		CD3DX12_ROOT_PARAMETER rootParameters[2];
		rootParameters[0].InitAsConstantBufferView(0);
		rootParameters[1].InitAsDescriptorTable(1, &texTable, D3D12_SHADER_VISIBILITY_PIXEL);

		// create a static sampler
		D3D12_STATIC_SAMPLER_DESC sampler = {};
		sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
		sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
		sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
		sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
		sampler.MipLODBias = 0;
		sampler.MaxAnisotropy = 0;
		sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
		sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
		sampler.MinLOD = 0.0f;
		sampler.MaxLOD = D3D12_FLOAT32_MAX;
		sampler.ShaderRegister = 0;
		sampler.RegisterSpace = 0;
		sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

		CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc;
		rootSignatureDesc.Init(_countof(rootParameters), // we have 2 root parameters
							   rootParameters,			 // a pointer to the beginning of our root parameters array
							   1,						 // we have one sampler
							   &sampler,				 // pointer to our sampler
							   D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | // we can deny shader stages here for better performance
								   D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
								   D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
								   D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS);
								   //D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS);

		ID3DBlob *signature;
		hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, nullptr);
		if (FAILED(hr))
		{
			LOG_FATAL("Error serializing root signature, createRootSignature()");
			return false;
		}

		hr = device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&rootSignature));
		if (FAILED(hr))
		{
			LOG_FATAL("Error creating root signature, createRootSignature()");
			return false;
		}

		LOG_TRACE("DirectX12 root signature created");

		return true;
	}

	bool DXRenderer::createPipeline()
	{
		LOG_TRACE("Creating DirectX12 pipeline");

		HRESULT hr;
		// create vertex and pixel shaders

		// when debugging, we can compile the shader files at runtime.
		// but for release versions, we can compile the hlsl shaders
		// with fxc.exe to create .cso files, which contain the shader
		// bytecode. We can load the .cso files at runtime to get the
		// shader bytecode, which of course is faster than compiling
		// them at runtime

		// compile vertex shader
		ID3DBlob *vertexShader; // d3d blob for holding vertex shader bytecode
		ID3DBlob *errorBuff;	// a buffer holding the error data if any
		hr = D3DCompileFromFile(L"../../shaders/VertexShader.hlsl",
								nullptr,
								nullptr,
								"main",
								"vs_5_0",
								D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
								0,
								&vertexShader,
								&errorBuff);
		if (FAILED(hr))
		{
			LOG_FATAL("Error compiling vertex shader, createPipeline()");
			OutputDebugStringA((char *)errorBuff->GetBufferPointer());
			return false;
		}

		// fill out a shader bytecode structure, which is basically just a pointer
		// to the shader bytecode and the size of the shader bytecode
		D3D12_SHADER_BYTECODE vertexShaderBytecode = {};
		vertexShaderBytecode.BytecodeLength = vertexShader->GetBufferSize();
		vertexShaderBytecode.pShaderBytecode = vertexShader->GetBufferPointer();

		// compile pixel shader
		ID3DBlob *pixelShader;
		hr = D3DCompileFromFile(L"../../shaders/PixelShader.hlsl",
								nullptr,
								nullptr,
								"main",
								"ps_5_0",
								D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
								0,
								&pixelShader,
								&errorBuff);
		if (FAILED(hr))
		{
			LOG_FATAL("Error compiling pixel shader, createPipeline()");
			OutputDebugStringA((char *)errorBuff->GetBufferPointer());
			return false;
		}

		// fill out shader bytecode structure for pixel shader
		D3D12_SHADER_BYTECODE pixelShaderBytecode = {};
		pixelShaderBytecode.BytecodeLength = pixelShader->GetBufferSize();
		pixelShaderBytecode.pShaderBytecode = pixelShader->GetBufferPointer();

		// create input layout

		// The input layout is used by the Input Assembler so that it knows
		// how to read the vertex data bound to it.

		D3D12_INPUT_ELEMENT_DESC inputLayout[] =
			{
				{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
				{"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
				{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
				{"HASCOLTEX", 0, DXGI_FORMAT_R32G32B32_UINT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
				{"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
		};

		// fill out an input layout description structure
		D3D12_INPUT_LAYOUT_DESC inputLayoutDesc = {};

		// we can get the number of elements in an array by "sizeof(array) / sizeof(arrayElementType)"
		inputLayoutDesc.NumElements = sizeof(inputLayout) / sizeof(D3D12_INPUT_ELEMENT_DESC);
		inputLayoutDesc.pInputElementDescs = inputLayout;

		// create a pipeline state object (PSO)

		// In a real application, you will have many pso's. for each different shader
		// or different combinations of shaders, different blend states or different rasterizer states,
		// different topology types (point, line, triangle, patch), or a different number
		// of render targets you will need a pso

		// VS is the only required shader for a pso. You might be wondering when a case would be where
		// you only set the VS. It's possible that you have a pso that only outputs data with the stream
		// output, and not on a render target, which means you would not need anything after the stream
		// output.

		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};						// a structure to define a pso
		psoDesc.InputLayout = inputLayoutDesc;									// the structure describing our input layout
		psoDesc.pRootSignature = rootSignature;									// the root signature that describes the input data this pso needs
		psoDesc.VS = vertexShaderBytecode;										// structure describing where to find the vertex shader bytecode and how large it is
		psoDesc.PS = pixelShaderBytecode;										// same as VS but for pixel shader
		psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; // type of topology we are drawing
		psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;						// format of the render target
		psoDesc.SampleDesc = sampleDesc;										// must be the same sample description as the swapchain and depth/stencil buffer
		psoDesc.SampleMask = 0xffffffff;										// sample mask has to do with multi-sampling. 0xffffffff means point sampling is done
		psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);		// a default rasterizer state.
		psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);					// a default blent state.
		psoDesc.NumRenderTargets = 1;											// we are only binding one render target
		psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);	// a default depth stencil state
		psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;

		// create the pso
		hr = device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pipelineStateObject));
		if (FAILED(hr))
		{
			LOG_FATAL("Error creating pipeline state object, createPipeline()");
			return false;
		}

		LOG_TRACE("DirectX12 pipeline created");

		return true;
	}

	bool DXRenderer::createCommandList()
	{
		LOG_TRACE("Creating DirectX12 command list");

		HRESULT hr;

		hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator[frameIndex], NULL, IID_PPV_ARGS(&commandList));
		if (FAILED(hr))
		{
			LOG_FATAL("Error creating command list, createCommandList()");
			return false;
		}

		LOG_TRACE("DirectX12 command list created");

		return true;
	}

	bool DXRenderer::createFencesAndEvents()
	{
		LOG_TRACE("Creating DirectX12 fences and events");
		HRESULT hr;
		// -- Create a Fence & Fence Event -- //

		// create the fences
		for (int i = 0; i < frameBufferCount; i++)
		{
			hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence[i]));
			if (FAILED(hr))
			{
				LOG_FATAL("Error creating fence, createFencesAndEvents()");
				return false;
			}
			fenceValue[i] = 0; // set the initial fence value to 0
		}

		// create a handle to a fence event
		fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
		if (fenceEvent == nullptr)
		{
			LOG_FATAL("Error creating fence event, createFencesAndEvents()");
			return false;
		}

		LOG_TRACE("DirectX12 fences and events created");

		return true;
	}

	bool DXRenderer::createBuffers(bool createDepthBufferOnly)
	{
		HRESULT hr;

		// Create the depth/stencil buffer

		// create a depth stencil descriptor heap so we can get a pointer to the depth stencil buffer
		D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
		dsvHeapDesc.NumDescriptors = 1;
		dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
		dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		hr = device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&dsDescriptorHeap));
		if (FAILED(hr))
		{
		}

		D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc = {};
		depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
		depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
		depthStencilDesc.Flags = D3D12_DSV_FLAG_NONE;

		D3D12_CLEAR_VALUE depthOptimizedClearValue = {};
		depthOptimizedClearValue.Format = DXGI_FORMAT_D32_FLOAT;
		depthOptimizedClearValue.DepthStencil.Depth = 1.0f;
		depthOptimizedClearValue.DepthStencil.Stencil = 0;

		int Width, Height;
		window->getSize(Width, Height);

		CD3DX12_HEAP_PROPERTIES heapPropertiesDefault(D3D12_HEAP_TYPE_DEFAULT);
		CD3DX12_RESOURCE_DESC depthStencilResourceDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, Width, Height, 1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
		device->CreateCommittedResource(
			&heapPropertiesDefault,
			D3D12_HEAP_FLAG_NONE,
			&depthStencilResourceDesc,
			D3D12_RESOURCE_STATE_DEPTH_WRITE,
			&depthOptimizedClearValue,
			IID_PPV_ARGS(&depthStencilBuffer));
		dsDescriptorHeap->SetName(L"Depth/Stencil Resource Heap");

		device->CreateDepthStencilView(depthStencilBuffer, &depthStencilDesc, dsDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

		// create the constant buffer resource heap
		// We will update the constant buffer one or more times per frame, so we will use only an upload heap
		// unlike previously we used an upload heap to upload the vertex and index data, and then copied over
		// to a default heap. If you plan to use a resource for more than a couple frames, it is usually more
		// efficient to copy to a default heap where it stays on the gpu. In this case, our constant buffer
		// will be modified and uploaded at least once per frame, so we only use an upload heap

		// first we will create a resource heap (upload heap) for each frame for the cubes constant buffers
		// As you can see, we are allocating 64KB for each resource we create. Buffer resource heaps must be
		// an alignment of 64KB. We are creating 3 resources, one for each frame. Each constant buffer is
		// only a 4x4 matrix of floats in this tutorial. So with a float being 4 bytes, we have
		// 16 floats in one constant buffer, and we will store 2 con stant buffers in each
		// heap, one for each cube, thats only 64x2 bits, or 128 bits we are using for each
		// resource, and each resource must be at least 64KB (65536 bits)
		for (int i = 0; i < frameBufferCount; ++i)
		{
			// create resource for cube 1
			CD3DX12_HEAP_PROPERTIES heapUpload(D3D12_HEAP_TYPE_UPLOAD);
			// If this memory is consumed fully, the app will crash
			CD3DX12_RESOURCE_DESC cbResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(1024 * 64 * 64); 
			hr = device->CreateCommittedResource(
				&heapUpload,					   // this heap will be used to upload the constant buffer data
				D3D12_HEAP_FLAG_NONE,			   // no flags
				&cbResourceDesc,				   // size of the resource heap. Must be a multiple of 64KB for single-textures and constant buffers
				D3D12_RESOURCE_STATE_GENERIC_READ, // will be data that is read from so we keep it in the generic read state
				nullptr,						   // we do not have use an optimized clear value for constant buffers
				IID_PPV_ARGS(&constantBufferUploadHeaps[i]));
			constantBufferUploadHeaps[i]->SetName(L"Constant Buffer Upload Resource Heap");

			ZeroMemory(&cbPerObject, sizeof(cbPerObject));

			CD3DX12_RANGE readRange(0, 0); // We do not intend to read from this resource on the CPU. (so end is less than or equal to begin)

			// map the resource heap to get a gpu virtual address to the beginning of the heap
			hr = constantBufferUploadHeaps[i]->Map(0, &readRange, reinterpret_cast<void **>(&cbvGPUAddress[i]));

			// Because of the constant read alignment requirements, constant buffer views must be 256 bit aligned. Our buffers are smaller than 256 bits,
			// so we need to add spacing between the two buffers, so that the second buffer starts at 256 bits from the beginning of the resource heap.
			memcpy(cbvGPUAddress[i], &cbPerObject, sizeof(cbPerObject));									  // cube1's constant buffer data
			memcpy(cbvGPUAddress[i] + ConstantBufferPerObjectAlignedSize, &cbPerObject, sizeof(cbPerObject)); // cube2's constant buffer data
		}

		finishedRecordingCommandList();

		executeCommandList();
 

		incrementFenceAndSignalCurrentFrame();

		return true;
	}

	void DXRenderer::createTextureDescriptorHeap(D3D12_DESCRIPTOR_HEAP_DESC heapDesc, ID3D12DescriptorHeap** descriptorHeap) {
		HRESULT hr;
		THROW_IF_FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(descriptorHeap)));
	}
	void DXRenderer::createTextureBuffer(ID3D12Resource** textureBuffer, ID3D12DescriptorHeap** descriptorHeap, D3D12_RESOURCE_DESC* textureDesc, BYTE* imageData, int bytesPerRow, TextureType texType) {
		HRESULT hr;
		
		resetCommandList();

		CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
		THROW_IF_FAILED(device->CreateCommittedResource(
			&heapProps,
			D3D12_HEAP_FLAG_NONE,
			textureDesc,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(textureBuffer)
		));

		(*textureBuffer)->SetName(L"Texture Buffer Resource Heap");

		UINT64 uploadBufferSize = 0;
		device->GetCopyableFootprints(textureDesc, 0, 1, 0, nullptr, nullptr, nullptr, &uploadBufferSize);

		CD3DX12_HEAP_PROPERTIES uploadHeapProps(D3D12_HEAP_TYPE_UPLOAD);
		CD3DX12_RESOURCE_DESC uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize);
		ID3D12Resource* textureBufferUploadHeap;
		THROW_IF_FAILED(device->CreateCommittedResource(
			&uploadHeapProps,
			D3D12_HEAP_FLAG_NONE,
			&uploadBufferDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&textureBufferUploadHeap)
		));
		textureBufferUploadHeap->SetName(L"Texture Buffer Upload Resource Heap");

		D3D12_SUBRESOURCE_DATA textureData{};
		ZeroMemory(&textureData, sizeof(textureData));
		textureData.pData = imageData; // &imageData[0];
		textureData.RowPitch = bytesPerRow;
		textureData.SlicePitch = bytesPerRow * textureDesc->Height;

		UpdateSubresources(commandList, (*textureBuffer), textureBufferUploadHeap, 0, 0, 1, &textureData);

		auto rb = CD3DX12_RESOURCE_BARRIER::Transition((*textureBuffer), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		commandList->ResourceBarrier(1, &rb);

		D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
		ZeroMemory(&heapDesc, sizeof(heapDesc));
		heapDesc.NumDescriptors = 1;
		heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;



		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
		ZeroMemory(&srvDesc, sizeof(srvDesc));
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Format = textureDesc->Format;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MipLevels = 1;

		auto offset = (int)texType;
		auto srv_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle((*descriptorHeap)->GetCPUDescriptorHandleForHeapStart(), offset, srv_size);

		device->CreateShaderResourceView((*textureBuffer), &srvDesc, srvHandle);

		finishedRecordingCommandList();
		executeCommandList();
		incrementFenceAndSignalCurrentFrame();
	}

	void DXRenderer::createIndexBuffer(ID3D12Resource** buffer, D3D12_INDEX_BUFFER_VIEW* bufferView, UINT64 bufferSize, BYTE* indexData) {
		HRESULT hr;
		
		resetCommandList();

		CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
		CD3DX12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
		THROW_IF_FAILED(device->CreateCommittedResource(
			&heapProps,
			D3D12_HEAP_FLAG_NONE,
			&resourceDesc,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(buffer)
		));

		(*buffer)->SetName(L"Index buffer");

		CD3DX12_HEAP_PROPERTIES uploadHeapProps(D3D12_HEAP_TYPE_UPLOAD);
		CD3DX12_RESOURCE_DESC uploadResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
		ID3D12Resource* uploadHeapBuffer;
		THROW_IF_FAILED(device->CreateCommittedResource(
			&uploadHeapProps,
			D3D12_HEAP_FLAG_NONE,
			&uploadResourceDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&uploadHeapBuffer)
		));

		uploadHeapBuffer->SetName(L"Index buffer resource upload heap");

		D3D12_SUBRESOURCE_DATA indexResourceData{};
		ZeroMemory(&indexResourceData, sizeof(indexResourceData));
		indexResourceData.pData = indexData;
		indexResourceData.RowPitch = bufferSize;
		indexResourceData.SlicePitch = bufferSize;

		UpdateSubresources(commandList, (*buffer), uploadHeapBuffer, 0, 0, 1, &indexResourceData);

		auto bufferRB = CD3DX12_RESOURCE_BARRIER::Transition((*buffer), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
		commandList->ResourceBarrier(1, &bufferRB);

		bufferView->BufferLocation = (*buffer)->GetGPUVirtualAddress();
		bufferView->Format = DXGI_FORMAT_R32_UINT;
		bufferView->SizeInBytes = bufferSize;

		finishedRecordingCommandList();
		executeCommandList();
		incrementFenceAndSignalCurrentFrame();
	}

	void DXRenderer::createVertexBuffer(ID3D12Resource** buffer, D3D12_VERTEX_BUFFER_VIEW* bufferView, UINT64 bufferSize, BYTE* vertexData) {
		HRESULT hr;

		resetCommandList();

		CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
		CD3DX12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
		THROW_IF_FAILED(device->CreateCommittedResource(
			&heapProps,
			D3D12_HEAP_FLAG_NONE,
			&resourceDesc,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(buffer)
		));

		(*buffer)->SetName(L"Vertex buffer");

		CD3DX12_HEAP_PROPERTIES uploadHeapProps(D3D12_HEAP_TYPE_UPLOAD);
		CD3DX12_RESOURCE_DESC uploadResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
		ID3D12Resource* uploadHeap;
		THROW_IF_FAILED(device->CreateCommittedResource(
			&uploadHeapProps,
			D3D12_HEAP_FLAG_NONE,
			&uploadResourceDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&uploadHeap)
		));

		uploadHeap->SetName(L"Vertex buffer upload heap");

		D3D12_SUBRESOURCE_DATA vertexResourceData{};
		ZeroMemory(&vertexResourceData, sizeof(vertexResourceData));
		vertexResourceData.pData = vertexData;
		vertexResourceData.RowPitch = bufferSize;
		vertexResourceData.SlicePitch = bufferSize;

		UpdateSubresources(commandList, (*buffer), uploadHeap, 0, 0, 1, &vertexResourceData);

		CD3DX12_RESOURCE_BARRIER bufferRB = CD3DX12_RESOURCE_BARRIER::Transition((*buffer), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
		commandList->ResourceBarrier(1, &bufferRB);

		bufferView->BufferLocation = (*buffer)->GetGPUVirtualAddress();
		bufferView->StrideInBytes = sizeof(Vertex);
		bufferView->SizeInBytes = bufferSize;

		finishedRecordingCommandList();
		executeCommandList();
		incrementFenceAndSignalCurrentFrame();
	}

	void DXRenderer::Cleanup()
	{
		// wait for the gpu to finish all frames
		for (int i = 0; i < frameBufferCount; ++i)
		{
			frameIndex = i;
			WaitForPreviousFrame();
		}

		// get swapchain out of full screen before exiting
		BOOL fs = false;
		if (swapChain->GetFullscreenState(&fs, NULL))
			swapChain->SetFullscreenState(false, NULL);

		SAFE_RELEASE(device);
		SAFE_RELEASE(swapChain);
		SAFE_RELEASE(commandQueue);
		SAFE_RELEASE(rtvDescriptorHeap);
		SAFE_RELEASE(commandList);
		SAFE_RELEASE(depthStencilBuffer);
		SAFE_RELEASE(dsDescriptorHeap);

		for (int i = 0; i < frameBufferCount; ++i)
		{
			SAFE_RELEASE(renderTargets[i]);
			SAFE_RELEASE(commandAllocator[i]);
			SAFE_RELEASE(fence[i]);
		};
	}

	void DXRenderer::WaitForPreviousFrame()
	{
		HRESULT hr;

		// swap the current rtv buffer index so we draw on the correct buffer
		frameIndex = swapChain->GetCurrentBackBufferIndex();

		// if the current fence value is still less than "fenceValue", then we know the GPU has not finished executing
		// the command queue since it has not reached the "commandQueue->Signal(fence, fenceValue)" command
		if (fence[frameIndex]->GetCompletedValue() < fenceValue[frameIndex])
		{
			// we have the fence create an event which is signaled once the fence's current value is "fenceValue"
			hr = fence[frameIndex]->SetEventOnCompletion(fenceValue[frameIndex], fenceEvent);
			if (FAILED(hr))
			{
				THROW_IF_FAILED(hr);
			}

			// We will wait until the fence has triggered the event that it's current value has reached "fenceValue". once it's value
			// has reached "fenceValue", we know the command queue has finished executing
			WaitForSingleObject(fenceEvent, INFINITE);
		}

		// increment fenceValue for next frame
		fenceValue[frameIndex]++;
	}
}