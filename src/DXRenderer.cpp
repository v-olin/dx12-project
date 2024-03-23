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

	DXRenderer::DXRenderer(Window& window) :
		windowHandle(window.windowHandle),
		useWarpDevice(false),
		rtvDescriptorSize(0)
	{ }


	bool DXRenderer::InitD3D()
	{
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

		// dis do be correct i think?
	//	ImGui_ImplDX12_Init(pDevice.Get(), frameBufferCount,
	//		DXGI_FORMAT_R8G8B8A8_UNORM, srvHeap.Get(),
	//		srvHeap.Get()->GetCPUDescriptorHandleForHeapStart(),
	//		srvHeap.Get()->GetGPUDescriptorHandleForHeapStart());
		return true;
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



	void DXRenderer::createFactory()
	{

	}

	void DXRenderer::createDebugController()
	{

	}

	void DXRenderer::createDevice()
	{

	}

	void DXRenderer::createCommandQueue()
	{

	}

	void DXRenderer::createSwapChain()
	{

	}
	void DXRenderer::createDescriptorHeaps()
	{

	}

	void DXRenderer::createCommandAllocators()
	{

	}

	void DXRenderer::createRootSignature()
	{

	}

	void DXRenderer::createPipeline() {

	}
	
	void DXRenderer::createCommandList()
	{

	}

	void DXRenderer::createFencesAndEvents()
	{

	}
}