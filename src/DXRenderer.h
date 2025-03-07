#pragma once

#include "PathWin.h"
#include "Event.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include <dxcapi.h>
#include "Helper.h"

#include <string>
#include <vector>
#include <wrl.h>

#include "GraphicsAPI.h"
#include "Window.h"
#include "Vertex.h"

#include "DXVertexBuffer.h"
#include "DXIndexBuffer.h"

#include "NVTLASGenerator.h"
#include "NVShaderBindingTableGenerator.h"

#define ALIGN_256(size) (((size) + 255) & ~255)

namespace nv = nvidia;

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
		bool initRaytracingPipeline(RenderSettings& renderSettings, Scene& scene);

		void initGraphicsAPI() override;
		void setClearColor(const dx::XMFLOAT3& color) override;
		void setCullFace(bool enabled) override;
		void setDepthTest(bool enabled) override;
		void setDrawTriangleOutline(bool enabled) override;
		void setViewport(int x, int y, int width, int height) override;
		GraphicsAPIType getGraphicsAPIType() override { return GraphicsAPIType::DirectX12; };

		void Render(RenderSettings& renderSettings, Scene& scene); // execute the command list
		//void Update(RenderSettings& renderSettings); // update the game logic

		static DXRenderer* getInstance() {
			static DXRenderer instance;
			return &instance;
		}

		int getModelsDrawn() const noexcept { return modelsDrawn; }
		int getMeshesDrawn() const noexcept { return meshesDrawn; }

		bool raytracingIsSupported() const noexcept { return raytracingSupported; }

		void createTextureDescriptorHeap(D3D12_DESCRIPTOR_HEAP_DESC heapDesc, ID3D12DescriptorHeap** descriptorHeap);
		void createTextureBuffer(ID3D12Resource** textureBuffer, ID3D12DescriptorHeap** descriptorHeap, D3D12_RESOURCE_DESC* textureDesc, BYTE* imageData, int bytesPerRow, TextureType texType);
		void createIndexBuffer(ID3D12Resource** buffer, D3D12_INDEX_BUFFER_VIEW* bufferView, UINT64 bufferSize, BYTE* indexData);
		void createVertexBuffer(ID3D12Resource** buffer, D3D12_VERTEX_BUFFER_VIEW* bufferView, UINT64 bufferSize, BYTE* vertexData);

		void cleanup(); // release com ojects and clean up memory
		
		virtual void onEvent(Event& e) override;

		void setProcWordValues(ProcedualWorldSettings settings);

	private:

		DXRenderer();

		HWND hwnd;
		Window* window = nullptr;
		bool useWarpDevice = false; // ???
		bool resizeOnNextFrame = false;
		UINT resizedWidth = 0, resizedHeight = 0;
		bool raytracingSupported = false;

		int modelsDrawn = 0;
		int meshesDrawn = 0;

		#pragma region DirectX stuff
		// I changed the commandlist to a 4 and the device to a 5 to get raytracing stuff
		ID3D12GraphicsCommandList4* commandList; // a command list we can record commands into, then execute them to render the frame
		ID3D12Device5* device; // direct3d device
		ID3D12CommandQueue* commandQueue; // container for command lists

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
		ID3D12PipelineState* trianglePipelineStateObject; // pso containing a pipeline state for drawing triangles
		ID3D12PipelineState* linePipelineStateObject; // pso containing a pipeline state for drawing lines
		ID3D12RootSignature* rootSignature; // root signature defines data shaders will access
		D3D12_VIEWPORT viewport; // area that output from rasterizer will be stretched to.
		D3D12_RECT scissorRect; // the area to draw in. pixels outside that area will not be drawn onto

		// the total size of the buffer, and the size of each element (vertex)
		DXGI_SAMPLE_DESC sampleDesc{};
		ID3D12Resource* depthStencilBuffer; // This is the memory for our depth buffer. it will also be used for a stencil buffer in a later tutorial
		ID3D12DescriptorHeap* dsDescriptorHeap; // This is a heap for our depth/stencil buffer descriptor
		ID3D12DescriptorHeap* mainDescriptorHeap; // this heap will store the descripor to our constant buffer

		// Bloom stuff
		ID3D12PipelineState* postProcessPipelineStateObject; // pso containing a pipeline state for post processing
		ID3D12Resource* postProcessTarget[3];
		ID3D12DescriptorHeap* postProcessHeap;
		ID3D12RootSignature* postProcessRootSignature;
		ID3D10Blob* bloomCsShaderBlob;

		// Motion blur stuff
		ID3D12PipelineState* motionBlurPipelineStateObject; // pso containing a pipeline state for motion blur
		ID3D12Resource* motionBlurTarget[2];
		ID3D12DescriptorHeap* motionBlurHeap;
		ID3D12RootSignature* motionBlurRootSignature;
		ID3D10Blob* motionBlurCsShaderBlob;

		#pragma endregion

		#pragma region Constant buffers
		// The constant buffer can't be bigger than 256 bytes
		struct ConstantBufferPerObject {
			DirectX::XMFLOAT4X4 wvpMat; // 64 bytes

			DirectX::XMFLOAT4X4 normalMatrix; // 64 bytes
			DirectX::XMFLOAT4X4 modelViewMatrix; // 64 bytes
			DirectX::XMFLOAT4X4 viewInverse; // 64 bytes
			DirectX::XMFLOAT4X4 viewMat; // 64 bytes

			PointLight pointLights[3]; // 48 bytes
			int pointLightCount; // 4 bytes
			bool isProcWorld;
		};

		struct ConstantBufferPerMesh {
			float4 material_emmision;
			float4 material_color;
			bool hasTexCoord; // 2 bytes (i think)
			bool pad1[3];
			bool hasNormalTex; // 2 bytes (i think)
			bool pad2[3];
			bool hasShinyTex; // 2 bytes (i think)
			bool pad3[3];
			bool hasMetalTex;
			bool pad4[3];
			bool hasFresnelTex;
			bool pad5[3];
			bool hasEmisionTex;
			bool pad6[3];
			float material_shininess;
			float material_metalness;
			float material_fresnel;
			bool hasMaterial;

			//for proc world only, should maybe not be here...
			float stop_flat;
			float stop_interp;
			int colorTexIndex;
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
		int ConstantBufferPerObjectAlignedSize = ALIGN_256(sizeof(ConstantBufferPerObject));
		int ConstantBufferPerMeshAlignedSize =  ALIGN_256(sizeof(ConstantBufferPerMesh));

		ConstantBufferPerObject cbPerObject; // this is the constant buffer data we will send to the gpu 
											// (which will be placed in the resource we created above)
		ConstantBufferPerMesh cbPerMesh;
		ID3D12Resource* constantBufferUploadHeap; // this is the memory on the gpu where constant buffers for each frame will be placed
		UINT8* cbvGPUAddress; // this is a pointer to each of the constant buffer resource heaps
		#pragma endregion

		#pragma region Raytracing stuff
		
		const D3D12_HEAP_PROPERTIES defaultHeapProps = {
			D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0
		};

		const D3D12_HEAP_PROPERTIES	deafultUploadHeapProps = {
			D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0
		};

		struct AccelerationStructureBuffers {
			ID3D12Resource* pScratch;		// scratch memory for AS builder
			ID3D12Resource* pResult;		// ptr to finished AS
			ID3D12Resource* pInstanceDesc;	// hold matrices of instances
		};

		ID3D12Resource* blas; // storage for BLAS
		nv::NVTLASGenerator tlasGenerator;
		AccelerationStructureBuffers tlasBuffers;
		std::vector<std::pair<ID3D12Resource*, DirectX::XMMATRIX>> asInstances;
		ID3D12Resource* instancePropsBuffer;
		nv::NVShaderBindingTableGenerator sbtGenerator;

		struct InstanceProperties {
			dx::XMMATRIX objToWorld;
		};

		struct CameraConstantBuffer {
			dx::XMMATRIX currView;
			dx::XMMATRIX currProj;
			dx::XMMATRIX currViewInv;
			dx::XMMATRIX currProjInv;
			dx::XMMATRIX prevView;
			dx::XMMATRIX prevProj;
			dx::XMMATRIX prevViewInv;
			dx::XMMATRIX prevProjInv;
			float nearPlane;
			float farPlane;
			bool useTAA;
			bool pad1[3];
			bool useMotionBlur;
			bool pad2[3];
		};

		CameraConstantBuffer cameraBuffer;

		unsigned int rtxFrameCount = 0;
		
		ID3D12Resource* cameraConstantBuffer;
		const uint32_t cameraConstantBufferSize = ALIGN_256(sizeof(CameraConstantBuffer));

		struct DXRLight {
			float4 position;
			float4 color;
			float intensity = 1.0f;
		};

		struct LightConstantBuffer {
			//DXRLight lights[3];
			float4 position0;
			float4 position1;
			float4 position2;
			float4 colorIntense0;
			float4 colorIntense1;
			float4 colorIntense2;
			//int pointLightCount = 0;
		};

		ID3D12Resource* lightConstantBuffer;
		const uint32_t lightConstantBufferSize = ALIGN_256(sizeof(LightConstantBuffer));

		IDxcBlob* rayGenLib;
		ID3D12RootSignature* rayGenSign;
		IDxcBlob* missLib;
		ID3D12RootSignature* missSign;
		IDxcBlob* hitLib;
		ID3D12RootSignature* hitSign;
		IDxcBlob* shadowLib;
		ID3D12RootSignature* shadowSign;

		ID3D12StateObject* rtpipelinestate;
		ID3D12StateObjectProperties* rtpipelinestateprops;

		ID3D12Resource* rtoutputbuffer;
		ID3D12Resource* rtdepthbuffer;
		D3D12_RESOURCE_STATES rtdepthstate;
		ID3D12DescriptorHeap* rtSrvUavHeap;
		ID3D12DescriptorHeap* constHeap;
		ID3D12Resource* sbtStorage;

		ID3D12Resource* dxrDepthBuffer;

		struct MeshData {
			float4 material_color;
			float4 material_emmision;
			float4x4 normalMatrix;
			bool hasTexCoord; // 2 bytes (i think)
			bool pad1[3];
			bool hasNormalTex; // 2 bytes (i think)
			bool pad2[3];
			bool hasShinyTex; // 2 bytes (i think)
			bool pad3[3];
			bool hasMetalTex;
			bool pad4[3];
			bool hasFresnelTex;
			bool pad5[3];
			bool hasEmisionTex;
			bool pad6[3];
			float material_shininess;
			float material_metalness;
			float material_fresnel;
			float material_transparency;
			float material_ior;
			bool hasMaterial;
		};
		ID3D12Resource* meshDataBuffer;
		UINT numMeshes;

		bool checkRaytracingSupport();

		ID3D12Resource* createASBuffers(UINT64 buffSize, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initStates, const D3D12_HEAP_PROPERTIES* heapProps = nullptr);
		AccelerationStructureBuffers createBLASFromModel(std::shared_ptr<Model> model);
		void createTLASFromBLAS(const std::vector<std::pair<ID3D12Resource*, DirectX::XMMATRIX>>& models, bool updateOnly = false);
		bool createAccelerationStructures(Scene& scene);
		bool createRaytracingPipeline(RenderSettings& renderSettings);
		bool createRaytracingOutputBuffer();
		bool createShaderResourceHeap(Scene& scene);
		bool createRTBuffers();
		bool createShaderBindingTable(Scene& scene);
		bool createMeshDataBuffer(Scene& scene);

		IDxcBlob* compileShaderLibrary(LPCWSTR libname);
		ID3D12RootSignature* createRayGenSignature();
		ID3D12RootSignature* createMissSignature();
		ID3D12RootSignature* createHitSignature();
		void updateRTBuffers(RenderSettings& settings, Scene& scene);
		void updateMeshDataBuffers(Scene& scene);
		void updateTLAS(Scene& scene);
		void performRaytracingPass();

		#pragma endregion

		#pragma region Random compute pass
		
		ID3D12Resource* noiseTexture;
		ID3D12Resource* noiseTextureRTX;
		ID3D12RootSignature* noisePassRootSignature;
		ID3D12PipelineState* noisePassPipelineState;
		ID3D12DescriptorHeap* noiseUavHeap;
		ID3DBlob* noiseCSBlob;

		struct NoiseConstBuffer {
			unsigned int frameNr;
		};

		unsigned int currentRTFrame = 0;

		const uint32_t noiseConstBuffSize = ALIGN_256(sizeof(NoiseConstBuffer));
		ID3D12Resource* noiseCBuffer;

		struct PipelineStateStream
		{
			CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
			CD3DX12_PIPELINE_STATE_STREAM_CS CS;
		};

		bool createRandomTexture();
		bool createRandomComputePass();
		bool createNoiseConstBuffer();
		void performNoisePass();

		#pragma endregion

		#pragma region TAA pass

		struct TAAConstantBuffer {
			dx::XMMATRIX currView;
			dx::XMMATRIX currProj;
			dx::XMMATRIX currViewInv;
			dx::XMMATRIX currProjInv;
			dx::XMMATRIX prevView;
			dx::XMMATRIX prevProj;
			dx::XMMATRIX prevViewInv;
			dx::XMMATRIX prevProjInv;
			float nearPlane;
			float farPlane;
			bool useTAA;
			bool pad1[3];
			bool useMotionBlur;
			bool pad2[3];
		};

		const uint32_t taaConstBuffSize = ALIGN_256(sizeof(TAAConstantBuffer));
		ID3D12Resource* taaConstBuffer;

		struct TAAFrame {
			ID3D12Resource* texture;
			D3D12_RESOURCE_STATES currState;
		};

		TAAFrame taaOutputFrame;
		TAAFrame historyBuffer;
		TAAFrame currentFrame;
		TAAFrame taadepthBuffer;
		bool taaUsedLastFrame = false;
		ID3D12RootSignature* taaPassRootSignature;
		ID3D12PipelineState* taaPassPipelineState;
		ID3D12DescriptorHeap* taaDescriptorHeap;
		ID3DBlob* taaCSBlob;

		bool createTAATextures();
		bool createTAAConstBuffer();
		bool createTAAComputePass();
		void performTAAPass(RenderSettings& renderSettings);
		void performBloomingEffect(RenderSettings& renderSettings);
		void performMotionBlur(RenderSettings& renderSettings);
		void transitionTAAFrame(TAAFrame& frame, D3D12_RESOURCE_STATES toState);
		void transitionResource(ID3D12Resource* resource, D3D12_RESOURCE_STATES fromState, D3D12_RESOURCE_STATES toState);

		#pragma endregion

		#pragma region Member functions
		void updatePipeline(RenderSettings& renderSettings, Scene& scene); // update the direct3d pipeline (update command lists)
		void finishedRecordingCommandList();
		void executeCommandList();
		void resetCommandList();
		void waitForPreviousFrame(); // wait until gpu is finished with command list
		void incrementFenceAndSignalCurrentFrame();

		bool createFactory();
		bool createDebugController();
		bool createDevice();
		bool createCommandQueue();
		bool createSwapChain();
		bool createDescriptorHeaps();
		bool createCommandAllocators();
		bool createRootSignature();
		bool createRasterPipeline();
		bool createLinePipeline();
		bool createPostProcessPipeline();
		bool createMotionBlurPipeline();
		bool createCommandList();
		bool createFencesAndEvents();
		bool createBuffers();

		bool onWindowResizeEvent(WindowResizeEvent& wre);
		void onResizeUpdatePipeline();
		void onResizeUpdateRenderTargets();
		void onResizeUpdateBackBuffers();
		void onResizeUpdateDescriptorHeaps();
		void waitForTotalGPUCompletion();

		// cleanup = cringe
		void destroyDevice();
		#pragma endregion

	};

}